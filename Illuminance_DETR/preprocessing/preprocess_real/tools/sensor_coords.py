"""
[7.5] センサ座標ファイル sensor_coords.json を書き出し、目視確認用の図を check/ に出す。

座標の定義（グリッド間隔・マット変位の当て方・正規化枠）は sensor_layout.py を参照。

  python sensor_coords.py --data oneperson_gym_01                    # 従来データ（legacy 枠）
  python sensor_coords.py --data normal --sensor_frame special
  python sensor_coords.py --data shift_03 --sensor_frame special --sensor_shift '[0.4, -0.2]'
  python sensor_coords.py --data rotate --sensor_frame special --sensor_rot '{"deg": 15, "pivot": [5, 5]}'

  # 既存の処理済みデータの移行（列名を ch 名に直し、sensor_coords.json を作る）:
  python sensor_coords.py --data oneperson_gym_01 oneperson_gym_02 ... --no-plot

light.csv / light_origin.csv / environment.csv の列名が旧形式（"gx,gy" や "0.4,1.0"）なら
ch 名に書き換える。値の行には触れず、ヘッダー行だけを差し替える。

check/ に出す図:
  sensor_layout.png : 上から見たセンサ配置。灰の白抜き = 変位なし（normal）、色付き = このデータ。
                      矢印は normal からの移動、★ は回転の支点、破線は正規化枠
  sensor_shadow.png : センサごとに「照度が下位 SHADOW_QUANTILE に入ったフレームの足首位置の中央値」
                      （× = 影の推定位置）と設定座標（●）を重ねた図。人が歩く範囲は全データで同じ
                      前提なので、歩行範囲の中心を normal のマット中心に合わせて描く。設定が正しければ
                      × は ● に、変位を当て忘れていれば × は灰の白抜きに寄る
"""
import os
import sys
import csv
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_ROOT
from sensor_layout import (COORDS_FILENAME, DEFAULT_FRAME, FRAMES, GRID_MAX, GRID_PITCH_M,
                           grid_of, parse_rot, parse_shift, positions_m,
                           rename_to_ch, to_normalized)

TS_FORMAT = "%Y-%m-%d %H:%M:%S %f"
SENSOR_FILES = ["light.csv", "light_origin.csv", "environment.csv"]

# 影の推定: センサごとに照度がこの分位以下のフレームを「影がかかった」とみなす
SHADOW_QUANTILE = 0.03
SHADOW_MIN_FRAMES = 20
# 照度と pose の時刻合わせの許容幅（秒）
MERGE_TOLERANCE_SEC = 0.1
# normal のマット中心（メートル）。pose は歩行範囲の中心が原点なので、ここへ平行移動して重ねる
MAT_CENTER_M = GRID_MAX * GRID_PITCH_M / 2


def rewrite_header(path, log_prefix):
    """ヘッダー行だけを ch 名に直す（値の行はバイト列のまま写す）。書き換えたら True。"""
    with open(path, newline="") as f:
        header_line = f.readline()
        header = next(csv.reader([header_line]))
        new_header, n_renamed = rename_to_ch(header, log_prefix)
        if not n_renamed:
            return False
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as out:
            csv.writer(out, quoting=csv.QUOTE_MINIMAL, lineterminator="\n").writerow(new_header)
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    os.replace(tmp, path)
    print(f"{log_prefix} {os.path.basename(path)}: 列名 {n_renamed} 個を ch 名に書き換えました")
    return True


def write_coords(data_dir, chs, shift, rot_deg, pivot, frame, log_prefix):
    pos = positions_m(chs, shift, rot_deg, pivot)
    norm = to_normalized(pos, frame, log_prefix)
    payload = {
        "grid_pitch_m": GRID_PITCH_M,
        "frame_m": {"name": frame, "origin": FRAMES[frame][0], "span": FRAMES[frame][1]},
        "sensor_shift_m": list(shift),
        "sensor_rot": {"deg": rot_deg, "pivot": list(pivot)} if pivot is not None else None,
        "sensors": {
            ch: {"grid": list(grid_of(ch)),
                 "m": [round(pos[ch][0], 6), round(pos[ch][1], 6)],
                 "norm": [round(norm[ch][0], 6), round(norm[ch][1], 6)]}
            for ch in chs
        },
    }
    path = os.path.join(data_dir, COORDS_FILENAME)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    us = [u for u, _ in norm.values()]
    vs = [v for _, v in norm.values()]
    print(f"{log_prefix} {COORDS_FILENAME} を書き出しました（{len(chs)} センサ, frame={frame}, "
          f"shift={list(shift)} m, rot={rot_deg}° pivot={pivot}, "
          f"norm x {min(us):.3f}〜{max(us):.3f} / y {min(vs):.3f}〜{max(vs):.3f}）")
    return pos


def _setup_axes(ax, title, frame):
    origin, span = FRAMES[frame]
    ax.add_patch(plt.Rectangle((origin, origin), span, span, fill=False, ls="--", lw=0.8, color="0.6"))
    ax.text(origin + 0.03, origin + span - 0.03, f"normalisation frame ({frame})", fontsize=7,
            color="0.5", va="top")
    # 軸の範囲はどちらの枠でも special の枠まで見せる（変位の大きさを同じ縮尺で比べられるように）
    lo, hi = min(origin, FRAMES["special"][0]), max(origin + span, sum(FRAMES["special"]))
    ax.annotate("camera", xy=(MAT_CENTER_M, lo + 0.05), xytext=(MAT_CENTER_M, lo - 0.25),
                ha="center", fontsize=8, arrowprops=dict(arrowstyle="->", color="0.3"),
                annotation_clip=False)
    ax.set_xlim(lo - 0.1, hi + 0.1)
    ax.set_ylim(lo - 0.35, hi + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]  (right, seen from camera)")
    ax.set_ylabel("y [m]  (far side)")
    ax.set_title(title, fontsize=10)
    ax.grid(lw=0.3, alpha=0.5)


def plot_layout(check_dir, data_name, pos, base, shift, rot_deg, pivot, frame):
    fig, ax = plt.subplots(figsize=(7, 7.4))
    _setup_axes(ax, f"{data_name}: sensor layout  (shift={list(shift)} m, rot={rot_deg}°)", frame)
    moved = any(np.hypot(pos[ch][0] - base[ch][0], pos[ch][1] - base[ch][1]) > 1e-9 for ch in pos)
    ax.scatter([p[0] for p in base.values()], [p[1] for p in base.values()], s=110,
               facecolors="none", edgecolors="0.6", label="no displacement (normal)")
    ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()], s=110,
               color="tab:blue", alpha=0.8, label=data_name)
    for ch, (x, y) in pos.items():
        ax.text(x, y + 0.09, ch, fontsize=6, ha="center", color="tab:blue")
        if moved:
            bx, by = base[ch]
            ax.annotate("", xy=(x, y), xytext=(bx, by),
                        arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.7))
    if pivot is not None:
        ax.scatter([pivot[0] * GRID_PITCH_M], [pivot[1] * GRID_PITCH_M], marker="*", s=320,
                   color="gold", edgecolors="k", zorder=5, label=f"pivot grid {list(pivot)}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(check_dir, "sensor_layout.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  → {out}")


def _foot_positions(pose_df):
    """1 人だけ写っているフレームの足首中点（無ければ腰中点）を (timestamp_dt, fx, fy) で返す。"""
    counts = pose_df.groupby("timestamp")["timestamp"].transform("size")
    df = pose_df[counts == 1].copy()
    fx = (df["left_ankle_X"] + df["right_ankle_X"]) / 2
    fy = (df["left_ankle_Y"] + df["right_ankle_Y"]) / 2
    hx = (df["left_hip_X"] + df["right_hip_X"]) / 2
    hy = (df["left_hip_Y"] + df["right_hip_Y"]) / 2
    out = pd.DataFrame({
        "timestamp_dt": pd.to_datetime(df["timestamp"], format=TS_FORMAT),
        "fx": fx.fillna(hx) + MAT_CENTER_M,
        "fy": fy.fillna(hy) + MAT_CENTER_M,
    }).dropna()
    return out.sort_values("timestamp_dt")


def plot_shadow(data_dir, check_dir, data_name, pos, base, frame):
    pose_path = os.path.join(data_dir, "pose.csv")
    if not os.path.exists(pose_path):
        print("  [skip] pose.csv が無いので sensor_shadow.png は作りません")
        return
    feet = _foot_positions(pd.read_csv(pose_path))
    if len(feet) < SHADOW_MIN_FRAMES:
        print("  [skip] 1 人だけ写ったフレームが少ないので sensor_shadow.png は作りません")
        return
    light = pd.read_csv(os.path.join(data_dir, "light.csv"))
    light["timestamp_dt"] = pd.to_datetime(light["timestamp"], format=TS_FORMAT)
    merged = pd.merge_asof(feet, light.sort_values("timestamp_dt"), on="timestamp_dt",
                           direction="nearest", tolerance=pd.Timedelta(seconds=MERGE_TOLERANCE_SEC))
    chs = [ch for ch in pos if ch in merged.columns]
    merged = merged.dropna(subset=chs)

    est = {}
    for ch in chs:
        v = merged[ch].astype(float)
        sel = merged[v <= v.quantile(SHADOW_QUANTILE)]
        if len(sel) >= SHADOW_MIN_FRAMES:
            est[ch] = (float(sel["fx"].median()), float(sel["fy"].median()))

    fig, ax = plt.subplots(figsize=(7, 7.4))
    _setup_axes(ax, f"{data_name}: shadow position (x) vs configured sensor (o)", frame)
    ax.scatter(merged["fx"], merged["fy"], s=1, color="0.85", alpha=0.3, zorder=0,
               label="foot trajectory")
    ax.scatter([p[0] for p in base.values()], [p[1] for p in base.values()], s=110,
               facecolors="none", edgecolors="0.6", label="no displacement (normal)")
    cmap = plt.get_cmap("tab20")
    for i, ch in enumerate(pos):
        color = cmap(i % 20)
        ax.scatter([pos[ch][0]], [pos[ch][1]], s=60, color=color)
        if ch in est:
            ex, ey = est[ch]
            ax.scatter([ex], [ey], marker="x", s=60, color=color, lw=2)
            ax.plot([pos[ch][0], ex], [pos[ch][1], ey], color=color, lw=0.8)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(check_dir, "sensor_shadow.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  → {out}（{len(est)}/{len(chs)} センサで推定, {len(merged)} フレーム）")

    # 図を開かなくても当たりが付くよう、推定位置との残差の中央値を設定配置と normal 配置の両方で出す。
    # 変位の設定が合っていれば「設定配置」側の残差が小さくなる。
    if est:
        for name, ref in (("configured", pos), ("normal", base)):
            dx = np.median([est[ch][0] - ref[ch][0] for ch in est])
            dy = np.median([est[ch][1] - ref[ch][1] for ch in est])
            print(f"  shadow - {name:<10}: median dx={dx:+.3f} m, dy={dy:+.3f} m")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="データフォルダ名（複数可）")
    parser.add_argument("--sensor_shift", default=None, metavar="JSON",
                        help="マットの平行移動 '[dx, dy]'（m、カメラから見て +x=右, +y=奥）")
    parser.add_argument("--sensor_rot", default=None, metavar="JSON",
                        help='マットの回転 \'{"deg": 15, "pivot": [5, 5]}\'（上から見て反時計回りが正、'
                             'pivot は支点センサのグリッド番号）')
    parser.add_argument("--sensor_frame", default=DEFAULT_FRAME, choices=sorted(FRAMES),
                        help=f"正規化枠（既定: {DEFAULT_FRAME}）。legacy = グリッド (0,0) が 0・2 m が 1（従来データ）、"
                             "special = [-0.6, 2.6] m（2026-10-09 の special 収録）。sensor_layout.py 参照")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                        help="check/ に sensor_layout.png / sensor_shadow.png を出すか（デフォルト: 出す）")
    args = parser.parse_args()

    try:
        shift = parse_shift(json.loads(args.sensor_shift) if args.sensor_shift else None)
        rot_deg, pivot = parse_rot(json.loads(args.sensor_rot) if args.sensor_rot else None)
    except (ValueError, json.JSONDecodeError) as e:
        sys.exit(f"[sensor] [error] {e}")

    for data_name in args.data:
        log_prefix = f"[sensor:{data_name}]"
        data_dir = os.path.join(PROCESSED_ROOT, data_name)
        if not os.path.exists(os.path.join(data_dir, "light.csv")):
            sys.exit(f"{log_prefix} [error] light.csv が見つかりません: {data_dir}")
        try:
            for name in SENSOR_FILES:
                path = os.path.join(data_dir, name)
                if os.path.exists(path):
                    rewrite_header(path, log_prefix)
            header = pd.read_csv(os.path.join(data_dir, "light.csv"), nrows=0).columns.tolist()
            chs = header[1:]
            pos = write_coords(data_dir, chs, shift, rot_deg, pivot, args.sensor_frame, log_prefix)
        except ValueError as e:
            sys.exit(f"[error] {e}")

        if not args.plot:
            continue
        # 図は検証用の付加物なので、失敗しても座標ファイルは残して先へ進む
        try:
            check_dir = os.path.join(data_dir, "check")
            os.makedirs(check_dir, exist_ok=True)
            base = positions_m(chs)
            plot_layout(check_dir, data_name, pos, base, shift, rot_deg, pivot, args.sensor_frame)
            plot_shadow(data_dir, check_dir, data_name, pos, base, args.sensor_frame)
        except Exception as e:
            print(f"{log_prefix} [warn] 確認図の作成に失敗しました（座標ファイルは作成済み）: {e}")


if __name__ == "__main__":
    main()
