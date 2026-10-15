"""
pose_origin.csv の person_id を RGB に重ねた ID 確認動画を作る（yaml の persons.ids を埋めるため）。

pose_origin.csv には 3D キーポイントしか無いので、intrinsics でカラー画像上に投影し直して
骨格と ID を描く（歪み係数は小さいのでピンホールで近似）。画面には次を出す:
  - 上部: 画像ファイル名（yaml の from / to にそのまま書ける）、start からの経過秒と
          phase_sec 秒単位のフェーズ枠番号（t // phase_sec % cycle_phases）
  - 下部: ID ごとの出現タイムライン（現在位置に縦線）。分断した ID がどこで切れて
          どこから始まるかを一目で追える

出力: processed/light/<データ名>/check/id_check.mp4

使い方:
  python make_id_video.py --data threeperson_gym_01
  python make_id_video.py --data oneperson_meeting_01 --cycle_phases 6
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_ROOT, raw_dir, processed_dir
from phase import timestamp_to_filename
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Illuminance_DETR.util.video import H264VideoWriter
from Illuminance_DETR.util.keypoints import RAW_SKELETON

JOINTS = sorted({j for pair in RAW_SKELETON for j in pair})
ROW_H = 12          # タイムライン 1 行の高さ(px)
LABEL_W = 44        # タイムライン左端の ID ラベル幅(px)


def id_color(pid):
    hue = (int(pid) * 47) % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def project(df, intr):
    """各関節の 3D 座標(m) を画素座標に投影した列 <joint>_u / <joint>_v を足す。"""
    for j in JOINTS:
        z = df[f"{j}_Z"]
        df[f"{j}_u"] = intr["fx"] * df[f"{j}_X"] / z + intr["ppx"]
        df[f"{j}_v"] = intr["fy"] * df[f"{j}_Y"] / z + intr["ppy"]
    return df


def draw_person(img, row, color):
    pts = {j: (row[f"{j}_u"], row[f"{j}_v"]) for j in JOINTS}
    pts = {j: (int(u), int(v)) for j, (u, v) in pts.items() if np.isfinite(u) and np.isfinite(v)}
    if not pts:
        return
    for a, b in RAW_SKELETON:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)
    xs, ys = zip(*pts.values())
    x1, y1, x2, y2 = min(xs) - 8, min(ys) - 8, max(xs) + 8, max(ys) + 8
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    label = str(int(row["person_id"]))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
    cv2.putText(img, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)


def build_timeline(present, ids, width):
    """ID x フレームの出現表を幅 width のバー画像にする（現在位置の線は毎フレーム描き足す）。"""
    n_frames = present.shape[1]
    img = np.full((ROW_H * len(ids) + 4, width, 3), 30, np.uint8)
    bar_w = width - LABEL_W
    cols = np.minimum((np.arange(n_frames) * bar_w / n_frames).astype(int), bar_w - 1)
    for r, pid in enumerate(ids):
        y0 = 2 + r * ROW_H
        on = np.zeros(bar_w, bool)
        on[cols[present[r]]] = True
        img[y0 + 2:y0 + ROW_H - 2, LABEL_W:][:, on] = id_color(pid)
        cv2.putText(img, str(pid), (4, y0 + ROW_H - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, id_color(pid), 1, cv2.LINE_AA)
    return img, cols


def header(width, text):
    img = np.zeros((26, width, 3), np.uint8)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def run(data_name, cycle_phases, fps, out_path=None, phase_sec=5.0):
    data_dir = processed_dir(data_name)
    pose = pd.read_csv(os.path.join(data_dir, "pose_origin.csv"))
    frames = pd.read_csv(os.path.join(data_dir, "timestamp.csv"))
    with open(os.path.join(raw_dir(data_name), "intrinsics.json")) as f:
        intr = json.load(f)

    # pose_origin.csv の範囲（sam3 なら start_img 以降）だけを動画にする
    frames = frames[(frames["timestamp"] >= pose["timestamp"].min()) &
                    (frames["timestamp"] <= pose["timestamp"].max())].reset_index(drop=True)
    pose = project(pose.dropna(subset=["person_id"]).copy(), intr)
    pose["person_id"] = pose["person_id"].astype(int)
    by_ts = {ts: g for ts, g in pose.groupby("timestamp")}

    ids = sorted(pose["person_id"].unique())
    frame_idx = {ts: i for i, ts in enumerate(frames["timestamp"])}
    present = np.zeros((len(ids), len(frames)), bool)
    for r, pid in enumerate(ids):
        idx = [frame_idx[t] for t in pose.loc[pose["person_id"] == pid, "timestamp"] if t in frame_idx]
        present[r, idx] = True

    t0 = pd.to_datetime(frames["timestamp"].iloc[0], format="%Y-%m-%d %H:%M:%S %f")
    elapsed = (pd.to_datetime(frames["timestamp"], format="%Y-%m-%d %H:%M:%S %f") - t0).dt.total_seconds()

    timeline, cols = None, None
    out_path = out_path or os.path.join(data_dir, "check", "id_check.mp4")
    writer = H264VideoWriter(out_path, fps)
    for i, row in tqdm(frames.iterrows(), total=len(frames), desc=data_name):
        img = cv2.imread(os.path.join(RAW_ROOT, row["color_filepath"]))
        if img is None:
            img = np.zeros((intr["height"], intr["width"], 3), np.uint8)
        if timeline is None:
            timeline, cols = build_timeline(present, ids, img.shape[1])

        g = by_ts.get(row["timestamp"])
        if g is not None:
            for _, p in g.iterrows():
                draw_person(img, p, id_color(p["person_id"]))
        t = elapsed.iloc[i]
        head = header(img.shape[1], f"{timestamp_to_filename(row['timestamp'])}   "
                                    f"t={t:6.1f}s   slot={int(t // phase_sec) % cycle_phases}")

        tl = timeline.copy()
        x = LABEL_W + cols[i]
        cv2.line(tl, (x, 0), (x, tl.shape[0] - 1), (255, 255, 255), 1)
        writer.write(np.vstack([head, img, tl]))
    writer.release()
    print(f"[id_video] {len(ids)} ids {[int(i) for i in ids]} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", required=True, help="データフォルダ名 例: threeperson_gym_01")
    parser.add_argument("--cycle_phases", type=int, default=5, help="1 サイクルのフェーズ数（meeting は 6）")
    parser.add_argument("--phase_sec", type=float, default=5.0, help="1 フェーズの秒数（小数可）")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default=None, help="出力先（既定: check/id_check.mp4）")
    args = parser.parse_args()
    run(args.data, args.cycle_phases, args.fps, args.out, args.phase_sec)


if __name__ == "__main__":
    main()
