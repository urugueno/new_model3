"""
カメラと照度センサーの時刻ずれ（shift値）を自動計算する。

手順:
1. light_origin.csv の全チャンネル平均が最初に閾値以下になった行を検出
2. color/ 画像の輝度平均が最初に閾値以下になったフレームを検出
3. shift = t_camera - t_light
   shift は照度側のタイムスタンプに「加算」される値。
   preprocess_real.py が light_ts + shift を行うので、これでカメラ時刻に揃う。

出力:
- check/shift.txt          : shift値（--shift 引数にそのまま渡せる形式）
- check/light_darkness.csv : 暗転判定行の前後7行
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_ROOT, processed_dir

LIGHT_THRESHOLD = 50.0   # 照度値がこれ以下で暗転と判定
IMAGE_THRESHOLD = 30.0   # 画像輝度平均がこれ以下で暗転と判定
CONTEXT_ROWS = 7         # 前後に取得する行数
SEARCH_SECONDS = 60      # 先頭から何秒間を暗転検出対象とするか


def detect_light_darkness(light_path: str):
    df = pd.read_csv(light_path)
    value_cols = [c for c in df.columns if c != "timestamp"]
    df["_mean"] = df[value_cols].mean(axis=1)
    df["_ts_sec"] = df["timestamp"].apply(ts_to_seconds)

    t_start = df["_ts_sec"].iloc[0]
    df_search = df[df["_ts_sec"] <= t_start + SEARCH_SECONDS]

    for idx in df_search.index:
        if df.at[idx, "_mean"] <= LIGHT_THRESHOLD:
            start = max(0, idx - CONTEXT_ROWS)
            end = min(len(df), idx + CONTEXT_ROWS + 1)
            context = df.iloc[start:end].drop(columns=["_mean", "_ts_sec"])
            return df.at[idx, "timestamp"], context

    return None, None


def detect_camera_darkness(ts_path: str):
    df_ts = pd.read_csv(ts_path)
    df_ts["_ts_sec"] = df_ts["timestamp"].apply(ts_to_seconds)
    t_start = df_ts["_ts_sec"].iloc[0]
    df_search = df_ts[df_ts["_ts_sec"] <= t_start + SEARCH_SECONDS]

    for _, row in df_search.iterrows():
        img_path = os.path.join(RAW_ROOT, row["color_filepath"])
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        brightness = np.mean(img)
        if brightness <= IMAGE_THRESHOLD:
            return row["timestamp"]

    return None


def ts_to_seconds(ts: str) -> float:
    parts = ts.strip().split(" ")
    date_part, time_part, micro_part = parts[0], parts[1], parts[2]
    h, m, s = time_part.split(":")
    total = int(h) * 3600 + int(m) * 60 + int(s) + int(micro_part) / 1e6
    return total


def seconds_to_shift_str(diff: float) -> str:
    sign = "-" if diff < 0 else ""
    diff = abs(diff)
    h = int(diff // 3600)
    m = int((diff % 3600) // 60)
    s = int(diff % 60)
    micro = round((diff - int(diff)) * 1e6)
    return f"{sign}{h:02d}{m:02d}{s:02d}.{micro:06d}"


def run(data_name: str, manual_shift: str = None):
    data_dir = processed_dir(data_name)
    check_dir = os.path.join(data_dir, "check")
    os.makedirs(check_dir, exist_ok=True)

    light_path = os.path.join(data_dir, "light_origin.csv")
    ts_path = os.path.join(data_dir, "timestamp.csv")

    if manual_shift:
        shift_path = os.path.join(check_dir, "shift.txt")
        with open(shift_path, "w") as f:
            f.write(manual_shift)
        print(f"[shift] 手動値を使用: {manual_shift}")
        return manual_shift

    print("[shift] 照度センサーの暗転を検出中...")
    t_light, light_context = detect_light_darkness(light_path)
    if t_light is None:
        print("[shift] [error] 照度センサーの暗転を検出できませんでした")
        sys.exit(1)
    print(f"  照度暗転タイムスタンプ: {t_light}")

    print("[shift] カラー画像の暗転を検出中...")
    t_camera = detect_camera_darkness(ts_path)
    if t_camera is None:
        print("[shift] [error] カメラ画像の暗転を検出できませんでした")
        sys.exit(1)
    print(f"  カメラ暗転タイムスタンプ: {t_camera}")

    # 照度クロックとカメラクロックの差。shift は照度側に加算される値なので符号を反転する
    # （light_ts + shift = camera_ts となるようにする）。
    # seconds_to_shift_str() は符号込みの文字列を返すため、ここで "-" を付け足さないこと。
    diff = ts_to_seconds(t_light) - ts_to_seconds(t_camera)
    shift_str = seconds_to_shift_str(-diff)
    print(f"  照度クロックはカメラより {-diff:+.6f} 秒ずれています")
    print(f"  shift = {shift_str} 秒")

    # shift.txt 保存
    shift_path = os.path.join(check_dir, "shift.txt")
    with open(shift_path, "w") as f:
        f.write(f"{shift_str}\n")
    print(f"  → check/shift.txt に保存しました")

    # light_darkness.csv 保存
    if light_context is not None:
        light_context.to_csv(os.path.join(check_dir, "light_darkness.csv"), index=False)
        print(f"  → check/light_darkness.csv に保存しました")

    print("[shift] 完了")
    return shift_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--shift", default=None, help="手動でshift値を指定する場合")
    args = parser.parse_args()
    run(args.data, args.shift)
