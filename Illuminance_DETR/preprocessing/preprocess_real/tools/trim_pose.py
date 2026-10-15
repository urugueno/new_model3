"""
pose_origin.csv から、指定した開始フレーム（しゃがみ始め）より前の行を削除する。
開始フレームはカラー画像のファイル名で指定する。

ファイル名形式: 20260520_064853_041393.jpg
→ タイムスタンプ: "2026-05-20 06:48:53 041393"
"""
import os
import sys
import argparse
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import processed_dir
# 位相の時刻原点と同じ変換を使う（relabel.py / preprocess_real.py と共有）
from phase import filename_to_timestamp

def run(data_name: str, start_img: str):
    data_dir = processed_dir(data_name)
    pose_path = os.path.join(data_dir, "pose_origin.csv")

    if not os.path.exists(pose_path):
        print(f"[error] pose_origin.csv が見つかりません: {pose_path}")
        sys.exit(1)

    start_ts = filename_to_timestamp(start_img)
    print(f"[trim] 開始タイムスタンプ: {start_ts}")

    df = pd.read_csv(pose_path)
    original_len = len(df)

    df = df[df["timestamp"] >= start_ts].copy()
    removed = original_len - len(df)

    if removed == 0:
        print("[trim] 削除対象の行はありませんでした")
    else:
        df.to_csv(pose_path, index=False)
        print(f"[trim] {removed} 行を削除しました（{original_len} → {len(df)} 行）")

    print("[trim] 完了")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--start_img", required=True, help="しゃがみ始めのカラー画像ファイル名 例: 20260520_064853_041393.jpg")
    args = parser.parse_args()
    run(args.data, args.start_img)
