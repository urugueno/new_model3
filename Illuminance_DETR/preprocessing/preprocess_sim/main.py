"""
シミュレーションデータ加工パイプライン エントリポイント

使い方:
  # real と同じ 6x6 グリッド（2x2 m の部屋）。real と混ぜて学習できる形式で出力する
  python main.py --data oneperson_sim_9Light_2x2_01 --fps 30 --date 20260716

  # それ以外の配置（4x4 / 6x6 m の部屋・ランダム配置）。従来形式で、real とは混ぜられない
  python main.py --data oneperson_sim_9Light_2x2_randomsensor_01 \
      --fps 30 --date 20260716 --max_coord 2.0

処理ステップ:
  [1]   preprocess_sim.py : pose.csv / light.csv /（あれば environment.csv, timestamp.csv）
                            および bbox_label.csv を生成
  [1.5] sensor_coords.py  : （real の tools/ のもの）sensor_coords.json と配置図を出力。
                            6x6 グリッドのときだけ
  [2]   relabel.py        : フレーム数ベースで行動ラベルを付与し bbox_label.csv を確定

real と混ぜて学習するため、6x6 グリッドのデータは出力を real と同じ形式にそろえる:
  - light.csv / environment.csv の列名はセンサ ID（ch0_pi2 など）、座標は sensor_coords.json
  - pose.csv は real と同じ関節列（COCO 名）だけ。座標は中心原点のメートル（real の --area_size 2.0 と同じ）
  - bbox は z1 = 0・z2 = 関節 Z の robust max（real の --fix-z と同じ）

real（01_preprocess_real）と異なり、offset計算・shift考慮・detect(YOLO)・重複確認・
poseのtrim は不要なため実行しない。
"""
import os
import sys
import csv
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_ROOT

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
# センサ座標の書き出しは real のツールをそのまま使う（座標の定義を 1 か所にするため）
REAL_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "preprocess_real", "tools")


def run_script(script: str, extra_args: list = None, tools_dir: str = TOOLS):
    cmd = [sys.executable, os.path.join(tools_dir, script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"  実行: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[error] {script} が失敗しました（終了コード: {result.returncode}）")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="シミュレーションデータ加工パイプライン",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data", required=True, help="データフォルダ名 例: oneperson_sim_...")
    parser.add_argument("--fps", type=int, required=True, help="pose のフレームレート 例: 30")
    parser.add_argument("--date", required=True, help="開始日 YYYYMMDD 例: 20260716")
    parser.add_argument("--max_coord", type=float, default=None,
                        help="real の 6x6 グリッドでない配置のときだけ必要。列名をこの値で割って"
                             "正規化座標にする（従来形式。real とは混ぜられない）")
    parser.add_argument("--sensor_frame", default=None, choices=["legacy", "special"],
                        help="sensor_coords.json の正規化枠。未指定なら legacy（real の従来データと同じ）。"
                             "混ぜる real データの枠に合わせること")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  パイプライン開始: {args.data}")
    print(f"{'#'*60}")

    # [1] preprocess: pose/light/timestamp/environment + bbox_label.csv
    preprocess_args = ["--data", args.data, "--fps", str(args.fps), "--date", args.date]
    if args.max_coord is not None:
        preprocess_args += ["--max_coord", str(args.max_coord)]
    run_script("preprocess_sim.py", preprocess_args)

    # [1.5] sensor_coords: 列名がセンサ ID（6x6 グリッド）のときだけ座標ファイルを書く
    with open(os.path.join(PROCESSED_ROOT, args.data, "light.csv"), newline="") as f:
        header = next(csv.reader(f))
    if header[1].startswith("ch"):
        sensor_args = ["--data", args.data]
        if args.sensor_frame:
            sensor_args += ["--sensor_frame", args.sensor_frame]
        run_script("sensor_coords.py", sensor_args, tools_dir=REAL_TOOLS)
    else:
        print("\n  → 従来形式（列名が座標）のため sensor_coords.json は作りません")

    # [2] relabel: フレーム数ベースのラベル付け・リネーム
    run_script("relabel.py", ["--data", args.data])

    print(f"\n{'#'*60}")
    print(f"  パイプライン完了: {args.data}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
