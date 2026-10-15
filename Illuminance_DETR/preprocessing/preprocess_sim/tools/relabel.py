"""
bbox_label.csv にフレーム数ベースの行動ラベルを付与し bbox_relabel.csv を生成する。
その後ファイルをリネームして最終成果物を bbox_label.csv にまとめる。

real（秒ベース）と異なり、sim は Blender の設定から算出したフレーム数でサイクルを区切る。
"""
import os
import sys
import shutil
import argparse
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import processed_dir

# --- 30fps換算のフレーム数設定 ---
# Blender設定(60fps)から半分にする
WALK_FIRST_30 = 157  # 314 / 2
WALK_NORMAL_30 = 165 # 330 / 2
STOP_TOTAL_30 = 465  # 930 / 2

# 「止まる」の中の3動作(静止、しゃがむ、手挙げ)を均等に割る
ACTION_STOP_30 = 155 # 465 / 3


def get_label_for_frame(frame_idx: int) -> int:
    """
    フレーム番号から行動ラベルを判定する。
    0=歩く, 1=静止, 2=しゃがむ, 3=両手挙げ
    """
    # 1サイクルの合計フレーム数 (歩き + 止まり全体)。初回だけ歩きが短い。
    first_cycle_total = WALK_FIRST_30 + STOP_TOTAL_30
    normal_cycle_total = WALK_NORMAL_30 + STOP_TOTAL_30

    if frame_idx < first_cycle_total:
        relative_f = frame_idx
        walk_limit = WALK_FIRST_30
    else:
        offset_f = frame_idx - first_cycle_total
        relative_f = offset_f % normal_cycle_total
        walk_limit = WALK_NORMAL_30

    if relative_f < walk_limit:
        return 0  # 歩く
    else:
        stop_f = relative_f - walk_limit
        if stop_f < ACTION_STOP_30:
            return 1  # 静止
        elif stop_f < ACTION_STOP_30 * 2:
            return 2  # しゃがむ
        else:
            return 3  # 両手挙げ


def run(data_name: str):
    data_dir = processed_dir(data_name)
    bbox_path = os.path.join(data_dir, "bbox_label.csv")
    relabel_path = os.path.join(data_dir, "bbox_relabel.csv")
    before_path = os.path.join(data_dir, "bbox_beforelabel.csv")

    if not os.path.exists(bbox_path):
        print(f"[relabel] [error] bbox_label.csv が見つかりません: {bbox_path}")
        sys.exit(1)

    df = pd.read_csv(bbox_path)
    # timestamp 昇順（フレーム順）に整列してから index でラベル付与
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["label"] = [get_label_for_frame(i) for i in range(len(df))]

    df.to_csv(relabel_path, index=False, float_format="%.6f")
    print(f"[relabel] bbox_relabel.csv を生成しました（{len(df)} フレーム）")
    print(f"[relabel] 設定: 歩き(初回 {WALK_FIRST_30}f / 通常 {WALK_NORMAL_30}f), "
          f"止まる3種(各 {ACTION_STOP_30}f)")

    # リネーム
    shutil.move(bbox_path, before_path)
    shutil.move(relabel_path, bbox_path)
    print(f"[relabel] bbox_label.csv → bbox_beforelabel.csv")
    print(f"[relabel] bbox_relabel.csv → bbox_label.csv")
    print("[relabel] 完了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    run(args.data)
