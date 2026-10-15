"""
pose_origin.csv から z_offset を自動計算し、check/offset_values.txt に出力する。

z_offset の意味:
  高さ変換 height = z_offset - camera_down（preprocess_real.py::process_pose_df）に用いる床基準。
  z_offset = median(各フレームの最大 camera_down) とすることで、bbox 床面 z1 の中央値が 0 になる。

水平方向(x/y)のオフセットは廃止（pose は --area_size で移動範囲にリスケールするため不要）。
bbox 座標グラフは preprocess_real.py 側（check/bbox_check.png）で出力する。
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import processed_dir


def calc_z_offset(df: pd.DataFrame) -> float:
    # camera_down は pose_origin.csv の _Y 列（Y↔Z 入替前）。各フレームの最大 = 床（足元）。
    y_cols = [c for c in df.columns if c.endswith("_Y")]
    per_frame_max_y = np.nanmax(df[y_cols].values, axis=1)
    return round(float(np.nanmedian(per_frame_max_y)), 6)


def run(data_name: str, manual_z=None):
    data_dir = processed_dir(data_name)
    check_dir = os.path.join(data_dir, "check")
    os.makedirs(check_dir, exist_ok=True)

    pose_path = os.path.join(data_dir, "pose_origin.csv")
    if not os.path.exists(pose_path):
        print(f"[offset] [error] pose_origin.csv が見つかりません: {pose_path}")
        sys.exit(1)

    df = pd.read_csv(pose_path)
    df = df.dropna(subset=["left_hip_X", "right_hip_X", "left_hip_Z", "right_hip_Z"])

    print("[offset] z_offset を計算中...")
    z_offset = manual_z if manual_z is not None else calc_z_offset(df)
    print(f"  z_offset = {z_offset}")

    values_path = os.path.join(check_dir, "offset_values.txt")
    with open(values_path, "w") as f:
        f.write(f"z_offset={z_offset}\n")
    print(f"  → check/offset_values.txt に保存しました")

    print("[offset] 完了")
    return z_offset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--z_offset", type=float, default=None)
    args = parser.parse_args()
    run(args.data, args.z_offset)
