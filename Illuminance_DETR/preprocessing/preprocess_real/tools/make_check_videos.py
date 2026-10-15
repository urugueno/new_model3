"""
前処理結果の目視検証用クリップを生成する。

処理済みディレクトリ配下に check/ を作り、録画全体からランダムに抽出した
検証用クリップを複数本生成する。各クリップは 1 フレームを 2x2 に分割し、

    +----------------+----------------+
    | RGB (raw jpg)  | 照度 6x6 grid  |
    +----------------+----------------+
    | pose 3D        | bbox 3D + label|
    +----------------+----------------+

を同一時刻で並べて描画する。これにより「しゃがんだ瞬間に pose が沈み、
bbox の高さが下がり、ラベルが crouch になり、真下のセンサーの照度が変化する」
といったモダリティ間のタイミング一致を人の目で確認できる。

時間軸は RGB 全フレーム基準（timestamp.csv の 30fps）。
- 照度は merge_asof(nearest) で対応付け、時刻差(ms)を数値表示する。
- pose / bbox は「ちょうど --people 人検出できたフレーム」しか存在しないため、
  該当時刻の行が無いフレームは "No data (filtered)" と表示する。
  （フィルタで落ちたフレームが見えること自体が検証情報になる）
- meeting データは お辞儀 のフェーズをフレームごと除外している（drop_phases）ため、
  "No data (filtered)" が規則的に現れる。3人データでは約半数のフレームが該当するが
  異常ではない。お辞儀の 5 秒区間がちょうど除外されているかの確認に使える。
  除外区間を跨いで見るには --duration 30（1 サイクル分）にするとよい。

クリップの選び方（--clips 本、デフォルト 5）:
- 1 本目は録画の先頭、最後の 1 本は録画の末尾で固定する。
  入りと終わり（トリム位置・開始フェーズ・最後まで検出が続いているか）は
  ランダム抽出では確認できないため。
- 残り (clips - 2) 本は中間を等分し、各区間からランダムに 1 本ずつ選ぶ。
  --seed で再現できる。

使い方:
  python make_check_videos.py --data threeperson_gym_01 --area_size 2.0
"""
import os
import sys
import random
import argparse

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D projection 登録のため必要)
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_ROOT, processed_dir
from sensor_layout import grid_of, rename_to_ch
# 共通の動画書き出し（H.264/mp4）。このスクリプトは main.py からサブプロセス起動され
# リポジトリルートが sys.path に無いので、ここで足してから読む。
# parents: [0]=tools [1]=preprocess_real [2]=preprocessing [3]=Illuminance_DETR [4]=repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Illuminance_DETR.util.video import H264VideoWriter
from Illuminance_DETR.util import keypoints

TS_FORMAT = "%Y-%m-%d %H:%M:%S %f"

# 出力レイアウト（各パネル PANEL_W x PANEL_H、全体は 2x2）
PANEL_W, PANEL_H = 640, 360
OUT_W, OUT_H = PANEL_W * 2, PANEL_H * 2

# 照度グリッド描画（visualize_map.py と同じ流儀）
LIGHT_CELL = 45
GRID_COLOR = (100, 100, 100)
COLOR_MAP = cv2.COLORMAP_PARULA

# relabel.py の phase_to_label に対応する現行 4 クラス
# （create_real.py の LABEL_MAP は旧 5 クラスなので使わない）
LABEL_MAP = {0: "walk", 1: "stop", 2: "crouch", 3: "hand"}

# 骨格は学習が実際に使う 14 点だけを描く（util/keypoints.py の定義をそのまま使う）。
# pose.csv には COCO 17 点が入っているが、目・耳・鼻は個別には描かず、
# 学習と同じく「取れている顔関節の平均」を head として 1 点にまとめて描く。
POSE_CONNECTIONS = keypoints.SKELETON

# 人物ごとの色（matplotlib 名 / cv2 BGR）
# 赤は "No data (filtered)" 表示専用にするため人物色には入れない
PERSON_COLORS = [
    ("tab:blue",   (200, 130,  40)),
    ("tab:orange", ( 20, 130, 240)),
    ("tab:green",  ( 60, 160,  60)),
    ("tab:purple", (180,  90, 150)),
    ("tab:brown",  ( 50,  90, 140)),
    ("tab:cyan",   (200, 190,  20)),
]

Z_MAX = 2.0            # 3D パネルの高さ方向の上限（メートル）
VIEW_ELEV, VIEW_AZIM = 18, -60


# --------------------------------------------------------------------------
# 読み込み
# --------------------------------------------------------------------------
def parse_ts(series):
    return pd.to_datetime(series, format=TS_FORMAT)


def load_inputs(data_name, bbox_override=None):
    """
    bbox_override を渡すと bbox_label.csv の代わりにその DataFrame を使う。
    ラベル付与前（CSV に書き出す前）の検証クリップを描くための口。
    """
    data_dir = processed_dir(data_name)
    paths = {
        "timestamp": os.path.join(data_dir, "timestamp.csv"),
        "light": os.path.join(data_dir, "light.csv"),
        "pose": os.path.join(data_dir, "pose.csv"),
        "bbox": os.path.join(data_dir, "bbox_label.csv"),
    }
    for key, path in paths.items():
        if key == "bbox" and bbox_override is not None:
            continue
        if not os.path.exists(path):
            print(f"[check] [error] {os.path.basename(path)} が見つかりません: {path}")
            sys.exit(1)

    ts_df = pd.read_csv(paths["timestamp"])
    light_df = pd.read_csv(paths["light"])
    pose_df = pd.read_csv(paths["pose"])
    bbox_df = bbox_override.copy() if bbox_override is not None else pd.read_csv(paths["bbox"])

    ts_df["timestamp_dt"] = parse_ts(ts_df["timestamp"])
    light_df["timestamp_dt"] = parse_ts(light_df["timestamp"])
    if "timestamp_dt" not in bbox_df.columns:
        bbox_df["timestamp_dt"] = parse_ts(bbox_df["timestamp"])

    ts_df = ts_df.sort_values("timestamp_dt").reset_index(drop=True)
    light_df = light_df.sort_values("timestamp_dt").reset_index(drop=True)

    return data_dir, ts_df, light_df, pose_df, bbox_df


def build_light_grid(light_df):
    """
    light.csv のセンサ列（ch 名）からセンサーのグリッド配置を構築する。
    マットをずらした収録でもマット上の並びは同じなので、sensor_coords.json の座標ではなく
    物理グリッド番号（sensor_layout.HEADER_MAP）で並べる。旧形式の "x,y" 列名も受け付ける。
    Y 反転は描画時に行う。
    """
    sensor_cols = [c for c in light_df.columns if c not in ("timestamp", "timestamp_dt")]
    try:
        ch_names, _ = rename_to_ch(["timestamp"] + sensor_cols, "[check] [error]")
    except ValueError as e:
        print(e)
        sys.exit(1)
    coords = [grid_of(ch) for ch in ch_names[1:]]
    n_x = max(c[0] for c in coords) + 1
    n_y = max(c[1] for c in coords) + 1

    values = light_df[sensor_cols].astype(float)
    vmin = float(values.min().min())
    vmax = float(values.max().max())
    return sensor_cols, coords, n_x, n_y, vmin, vmax


def build_person_colors(pose_df, bbox_df):
    """person_id -> 色 の対応を、出現順で決定的に割り当てる。"""
    ids = []
    for df in (pose_df, bbox_df):
        if "person_id" in df.columns:
            for pid in df["person_id"].tolist():
                if pd.notna(pid) and pid not in ids:
                    ids.append(pid)
    return {pid: PERSON_COLORS[i % len(PERSON_COLORS)] for i, pid in enumerate(ids)}


# --------------------------------------------------------------------------
# パネル描画
# --------------------------------------------------------------------------
def put_text(img, text, org, color=(255, 255, 255), scale=0.5, thickness=1):
    """黒縁付きで読みやすくテキストを描く。"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def darken_strip(img, y0, y1):
    """テキストを重ねる帯を暗くして可読性を上げる。"""
    img[y0:y1] = (img[y0:y1] * 0.35).astype(np.uint8)


def render_rgb_panel(color_filepath, ts_str, elapsed):
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    img = None
    if isinstance(color_filepath, str):
        full_path = os.path.join(RAW_ROOT, color_filepath)
        img = cv2.imread(full_path)
    if img is None:
        put_text(panel, "RGB: image missing", (16, PANEL_H // 2), (0, 0, 255), 0.7, 2)
    else:
        panel = cv2.resize(img, (PANEL_W, PANEL_H))
    darken_strip(panel, PANEL_H - 48, PANEL_H)
    put_text(panel, "RGB", (12, 26), (255, 255, 255), 0.7, 2)
    put_text(panel, ts_str, (12, PANEL_H - 34))
    # put_text(panel, f"t = {elapsed:+.3f} s", (12, PANEL_H - 12))
    return panel


def render_light_panel(row, sensor_cols, coords, dim_x, dim_y, vmin, vmax, dt_ms):
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    grid_w, grid_h = dim_x * LIGHT_CELL, dim_y * LIGHT_CELL
    ox, oy = 60, 40

    if row is None:
        put_text(panel, "Illuminance: No data", (16, PANEL_H // 2), (0, 0, 255), 0.7, 2)
        return panel

    span = vmax - vmin
    for i, (x, y) in enumerate(coords):
        value = float(row[sensor_cols[i]])
        clipped = max(vmin, min(vmax, value))
        norm = 0 if span <= 0 else int(255 * (clipped - vmin) / span)
        color = cv2.applyColorMap(np.array([[norm]], dtype=np.uint8), COLOR_MAP)[0, 0].tolist()
        cv_y = (dim_y - 1) - y  # visualize_map.py と同じ Y 反転（row 0 を下に）
        tl = (ox + x * LIGHT_CELL, oy + cv_y * LIGHT_CELL)
        br = (ox + (x + 1) * LIGHT_CELL, oy + (cv_y + 1) * LIGHT_CELL)
        cv2.rectangle(panel, tl, br, color, -1)

    for i in range(dim_x + 1):
        cv2.line(panel, (ox + i * LIGHT_CELL, oy), (ox + i * LIGHT_CELL, oy + grid_h), GRID_COLOR, 1)
    for i in range(dim_y + 1):
        cv2.line(panel, (ox, oy + i * LIGHT_CELL), (ox + grid_w, oy + i * LIGHT_CELL), GRID_COLOR, 1)

    # カラーバー（上が vmax）
    bar_x, bar_w = ox + grid_w + 30, 22
    for py in range(grid_h):
        norm = int(255 * (1 - py / max(grid_h - 1, 1)))
        color = cv2.applyColorMap(np.array([[norm]], dtype=np.uint8), COLOR_MAP)[0, 0].tolist()
        cv2.line(panel, (bar_x, oy + py), (bar_x + bar_w, oy + py), color, 1)
    cv2.rectangle(panel, (bar_x, oy), (bar_x + bar_w, oy + grid_h), GRID_COLOR, 1)
    put_text(panel, f"{vmax:.0f}", (bar_x + bar_w + 6, oy + 12))
    put_text(panel, f"{vmin:.0f}", (bar_x + bar_w + 6, oy + grid_h))

    put_text(panel, "light.csv", (12, 26), (255, 255, 255), 0.7, 2)
    put_text(panel, str(row["timestamp"]), (12, PANEL_H - 34))
    # RGB フレーム時刻との差。同期ズレが目視だけでなく数値でも分かる。
    # dt_color = (255, 255, 255) if abs(dt_ms) < 20 else (0, 200, 255)
    # put_text(panel, f"dt vs RGB = {dt_ms:+.1f} ms", (12, PANEL_H - 12), dt_color)
    return panel


def fig_to_bgr(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def setup_3d_axis(ax, area_size):
    """毎フレーム同じ視点・同じ軸範囲にする（視点が動くと目視比較できない）。"""
    half = area_size / 2
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(0, Z_MAX)
    ax.set_box_aspect((area_size, area_size, Z_MAX))
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xticks(np.arange(-half, half + 1e-6, max(half, 0.5)))
    ax.set_yticks(np.arange(-half, half + 1e-6, max(half, 0.5)))
    ax.set_zticks(np.arange(0, Z_MAX + 1e-6, 0.5))
    ax.set_xlabel("X [m]", fontsize=8, labelpad=-4)
    ax.set_ylabel("Y [m]", fontsize=8, labelpad=-4)
    ax.set_zlabel("Z [m]", fontsize=8, labelpad=-4)
    ax.tick_params(labelsize=7, pad=-2)


def draw_skeleton(ax, pose_row, color):
    """学習に渡るのと同じ 14 点（実測 12 + 合成 neck/head）を描く。色は人物ごと。"""
    points = keypoints.output_points(pose_row)

    for start_part, end_part in POSE_CONNECTIONS:
        if start_part in points and end_part in points:
            p1, p2 = points[start_part], points[end_part]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=1.6)
    if points:
        pts = np.array(list(points.values()))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=6)


def draw_bbox_3d(ax, x1, y1, z1, x2, y2, z2, color, label=None, label_lift=0.0):
    """create_real.py の draw_bbox_3d を流用。"""
    p = np.array([
        [x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],
        [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot([p[i][0], p[j][0]], [p[i][1], p[j][1]], [p[i][2], p[j][2]],
                color=color, linewidth=1.4)
    if label is not None:
        # 箱の上に出し、人物ごとに高さをずらして文字同士の重なりを避ける
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, z2 + 0.08 + label_lift, str(label),
                color=color, fontsize=9, ha="center", weight="bold")


def render_3d_panel(fig, ax, area_size, title, rows, person_colors, kind, bbox_text="label"):
    """bbox_text: "label" なら行動ラベル、"id" なら person_id（YAML の ids）を箱の上に出す。"""
    ax.cla()
    setup_3d_axis(ax, area_size)
    if rows is None or len(rows) == 0:
        ax.text2D(0.5, 0.5, "No data (filtered)", transform=ax.transAxes,
                  color="red", fontsize=14, ha="center", va="center")
        panel = fig_to_bgr(fig)
        cv2.putText(panel, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
        return panel

    for slot, (_, row) in enumerate(rows.iterrows()):
        pid = row.get("person_id", None)
        mpl_color = person_colors.get(pid, PERSON_COLORS[0])[0] if pd.notna(pid) else PERSON_COLORS[0][0]
        if kind == "pose":
            draw_skeleton(ax, row, mpl_color)
        else:
            label = int(row["label"]) if pd.notna(row["label"]) else -1
            if bbox_text == "id":
                pid_text = "?" if pd.isna(pid) else f"{int(pid)}"
                text = f"id={pid_text} {LABEL_MAP.get(label, '?')}"
            else:
                text = f"{label}:{LABEL_MAP.get(label, '?')}"
            draw_bbox_3d(ax, row["x1"], row["y1"], row["z1"],
                         row["x2"], row["y2"], row["z2"], mpl_color, text,
                         label_lift=0.14 * slot)
    panel = fig_to_bgr(fig)
    cv2.putText(panel, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    return panel


# --------------------------------------------------------------------------
# クリップ生成
# --------------------------------------------------------------------------
def pick_clip_starts(bbox_df, duration, clips, seed):
    """
    クリップの開始時刻を選ぶ。先頭と末尾は必ず含め、残りをランダムに散らす。

    - 1 本目: 録画の先頭（最初のフレームから）
    - 最後  : 録画の末尾（最終フレームで終わる）
    - 間の (clips - 2) 本: 中間区間を等分し、各区間内でランダムに 1 本
      （等分してから選ぶので、ランダムでも固まらず全体に散る）

    先頭・末尾を固定するのは、録画の入りと終わりが正しく処理できているか
    （トリム位置・ラベルの開始フェーズ・最後まで検出が続いているか）が
    ランダム抽出だと確認できないため。

    候補範囲は bbox の時間帯に限定する。timestamp.csv は trim_pose.py でトリムされる
    前の全フレームを含むため、限定しないと pose/bbox が空の区間を引いてしまう。
    """
    t_min = bbox_df["timestamp_dt"].min()
    t_max = bbox_df["timestamp_dt"].max()
    total = (t_max - t_min).total_seconds()
    if total <= 0:
        print("[check] [error] bbox_label.csv の時間範囲が不正です")
        sys.exit(1)

    usable = total - duration
    if usable <= 0:
        print(f"[check] [warn] データ長 {total:.1f}s が duration {duration}s 以下のため 1 本にまとめます")
        return [t_min]

    def at(offset_sec):
        return t_min + pd.to_timedelta(offset_sec, unit="s")

    if clips <= 1:
        return [at(0.0)]
    if clips == 2:
        return [at(0.0), at(usable)]

    n_random = clips - 2

    # ランダム分は先頭・末尾のクリップと重ならない内側の範囲から選ぶ。
    # 録画が短くて内側が取れない場合は全体から選ぶ（重複してもクリップ本数は保つ）。
    lo, hi = duration, usable - duration
    if hi <= lo:
        lo, hi = 0.0, usable

    rng = random.Random(seed)
    seg = (hi - lo) / n_random
    starts = [at(0.0)]
    for i in range(n_random):
        starts.append(at(lo + seg * i + rng.uniform(0, seg)))
    starts.append(at(usable))
    return starts


def make_clip(out_path, frames, light_df, sensor_cols, coords, dim_x, dim_y, vmin, vmax,
              pose_by_ts, bbox_by_ts, person_colors, area_size, fps,
              pose_fig, pose_ax, bbox_fig, bbox_ax, bbox_text="label", mark_ts=None,
              mark_note=""):
    # 照度は RGB 各フレームに対し nearest で対応付ける（RGB 30Hz / 照度 約88Hz）
    matched = pd.merge_asof(
        frames[["timestamp_dt"]].sort_values("timestamp_dt"),
        light_df[["timestamp_dt"]].assign(light_idx=np.arange(len(light_df))),
        on="timestamp_dt", direction="nearest",
    )
    light_idx = matched["light_idx"].to_numpy()

    writer = H264VideoWriter(out_path, fps, (OUT_W, OUT_H))
    if not writer.isOpened():
        print(f"[check] [error] VideoWriter を開けませんでした: {out_path}")
        return False

    t0 = frames["timestamp_dt"].iloc[0]
    try:
        for i, (_, frame) in enumerate(frames.iterrows()):
            ts_str = frame["timestamp"]
            elapsed = (frame["timestamp_dt"] - t0).total_seconds()

            if np.isnan(light_idx[i]):
                light_row, dt_ms = None, 0.0
            else:
                light_row = light_df.iloc[int(light_idx[i])]
                dt_ms = (light_row["timestamp_dt"] - frame["timestamp_dt"]).total_seconds() * 1000

            canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
            canvas[0:PANEL_H, 0:PANEL_W] = render_rgb_panel(frame["color_filepath"], ts_str, elapsed)
            canvas[0:PANEL_H, PANEL_W:OUT_W] = render_light_panel(
                light_row, sensor_cols, coords, dim_x, dim_y, vmin, vmax, dt_ms)
            canvas[PANEL_H:OUT_H, 0:PANEL_W] = render_3d_panel(
                pose_fig, pose_ax, area_size, "pose.csv", pose_by_ts.get(ts_str), person_colors, "pose")
            canvas[PANEL_H:OUT_H, PANEL_W:OUT_W] = render_3d_panel(
                bbox_fig, bbox_ax, area_size, "bbox_label.csv", bbox_by_ts.get(ts_str), person_colors, "bbox",
                bbox_text=bbox_text)

            cv2.line(canvas, (0, PANEL_H), (OUT_W, PANEL_H), (80, 80, 80), 2)
            cv2.line(canvas, (PANEL_W, 0), (PANEL_W, OUT_H), (80, 80, 80), 2)

            # 違反フレームは赤枠で囲む。前後 3 秒のクリップの中で
            # 「どの瞬間が壊れているか」を早送りせずに見つけられるようにする。
            if mark_ts and ts_str in mark_ts:
                cv2.rectangle(canvas, (2, 2), (OUT_W - 3, OUT_H - 3), (0, 0, 255), 6)
                put_text(canvas, mark_note or "VIOLATION", (PANEL_W - 170, OUT_H - 14),
                         (0, 0, 255), 0.9, 2)

            writer.write(canvas)
    finally:
        writer.release()
    return True


def render_windows(data_name, area_size, windows, out_prefix="clip", fps=30,
                   bbox_override=None, bbox_text="label", mark_ts=None, mark_note="",
                   log_prefix="[check]"):
    """
    時間窓 [(start, end), ...] ぶんのクリップを check/ に書き出す共通ルーチン。

    通常の検証クリップ（run）も、ラベル多重集合の違反クリップ（verify_labels）も
    ここを通す。描画の見た目が 2 箇所に分かれると、並べて比較する意味が薄れるため。

    bbox_override : bbox_label.csv の代わりに使う DataFrame（CSV 書き出し前の検証用）
    bbox_text     : bbox の上に出す文字 "label" / "id"
    mark_ts       : 赤枠で強調する timestamp 文字列の集合
    戻り値        : (check_dir, [{"path","start","frames","pose_frames"}...], meta)
    """
    data_dir, ts_df, light_df, pose_df, bbox_df = load_inputs(data_name, bbox_override)
    sensor_cols, coords, dim_x, dim_y, vmin, vmax = build_light_grid(light_df)
    person_colors = build_person_colors(pose_df, bbox_df)

    # timestamp 文字列で完全一致検索する（pose/bbox の timestamp は timestamp.csv 由来）
    pose_by_ts = {ts: g for ts, g in pose_df.groupby("timestamp")}
    bbox_by_ts = {ts: g for ts, g in bbox_df.groupby("timestamp")}

    check_dir = os.path.join(data_dir, "check")
    os.makedirs(check_dir, exist_ok=True)

    print(f"{log_prefix} 照度スケール: {vmin:.1f} 〜 {vmax:.1f} lx（全クリップ共通）")
    print(f"{log_prefix} グリッド: {dim_x} x {dim_y} / センサー {len(sensor_cols)} 個")

    # figure はループ外で 1 つずつ作って使い回す（毎フレーム生成すると極端に遅い）
    pose_fig = plt.figure(figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100)
    pose_ax = pose_fig.add_axes([0.02, 0.02, 0.96, 0.96], projection="3d")
    bbox_fig = plt.figure(figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100)
    bbox_ax = bbox_fig.add_axes([0.02, 0.02, 0.96, 0.96], projection="3d")

    results = []
    try:
        for i, (start, end) in enumerate(windows, start=1):
            name = f"{out_prefix}_{i:02d}"
            frames = ts_df[(ts_df["timestamp_dt"] >= start) & (ts_df["timestamp_dt"] < end)]
            if frames.empty:
                print(f"{log_prefix} [warn] {name}: 対象フレームが 0 のためスキップします")
                continue

            out_path = os.path.join(check_dir, f"{name}.mp4")
            n_pose = sum(1 for ts in frames["timestamp"] if ts in pose_by_ts)
            print(f"{log_prefix} {name}: {frames['timestamp'].iloc[0]} から "
                  f"{len(frames)} フレーム（pose あり {n_pose}）")

            with tqdm(total=len(frames), desc=name, leave=False) as bar:
                ok = make_clip(out_path, frames, light_df, sensor_cols, coords, dim_x, dim_y,
                               vmin, vmax, pose_by_ts, bbox_by_ts, person_colors, area_size, fps,
                               pose_fig, pose_ax, bbox_fig, bbox_ax,
                               bbox_text=bbox_text, mark_ts=mark_ts, mark_note=mark_note)
                bar.update(len(frames))
            if ok:
                results.append({"path": out_path, "name": name,
                                "start": frames["timestamp"].iloc[0],
                                "frames": len(frames), "pose_frames": n_pose})
    finally:
        plt.close(pose_fig)
        plt.close(bbox_fig)

    meta = {"vmin": vmin, "vmax": vmax, "dim_x": dim_x, "dim_y": dim_y,
            "sensors": len(sensor_cols)}
    return check_dir, results, meta


def run(data_name, area_size, clips, duration, fps, seed):
    data_dir = processed_dir(data_name)
    bbox_path = os.path.join(data_dir, "bbox_label.csv")
    if not os.path.exists(bbox_path):
        print(f"[check] [error] bbox_label.csv が見つかりません: {bbox_path}")
        sys.exit(1)

    # 抽出タイミングを決めるために時刻範囲だけ先に読む
    bbox_df = pd.read_csv(bbox_path)
    bbox_df["timestamp_dt"] = parse_ts(bbox_df["timestamp"])

    starts = pick_clip_starts(bbox_df, duration, clips, seed)
    windows = [(s, s + pd.to_timedelta(duration, unit="s")) for s in starts]
    print(f"[check] {len(windows)} 本のクリップを生成します（各 {duration}s @ {fps}fps, seed={seed}）")
    if len(windows) >= 2:
        print(f"[check] 内訳: 先頭 1 本 + ランダム {len(windows) - 2} 本 + 末尾 1 本")

    check_dir, results, meta = render_windows(data_name, area_size, windows, fps=fps)

    log_lines = [
        f"data={data_name}",
        f"seed={seed} clips={clips} duration={duration}s fps={fps} area_size={area_size}",
        f"light_scale={meta['vmin']:.3f}..{meta['vmax']:.3f}",
        "",
    ]
    for r in results:
        log_lines.append(f"{r['name']}.mp4  start={r['start']}  "
                         f"frames={r['frames']}  pose_frames={r['pose_frames']}")

    with open(os.path.join(check_dir, "clips.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"[check] 完了: {len(results)} 本を生成しました -> {check_dir}")
    return len(results)


def main():
    parser = argparse.ArgumentParser(
        description="前処理結果（照度/RGB/pose/bbox）の目視検証用クリップを生成する",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", required=True, help="データフォルダ名 例: threeperson_gym_01")
    parser.add_argument("--area_size", type=float, default=2.0,
                        help="3D パネルの軸範囲（メートル）。前処理の --area_size と同じ値にする")
    parser.add_argument("--clips", type=int, default=5,
                        help="生成するクリップ本数（デフォルト: 5）。"
                             "先頭 1 本と末尾 1 本は必ず含み、残りをランダムに選ぶ")
    parser.add_argument("--duration", type=float, default=10.0, help="1 本の長さ・秒（デフォルト: 10）")
    parser.add_argument("--fps", type=int, default=30, help="出力 fps（デフォルト: 30）")
    parser.add_argument("--seed", type=int, default=0, help="抽出タイミングの乱数シード（再現性確保）")
    args = parser.parse_args()

    run(args.data, args.area_size, args.clips, args.duration, args.fps, args.seed)


if __name__ == "__main__":
    main()
