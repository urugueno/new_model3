"""
バッチ実行スクリプト（YAML 設定で複数データをまとめて処理）

使い方:
  python main_batch.py batch/my_run.yaml

設定ファイル（YAML）の書き方:
  defaults に共通設定、jobs にデータごとの設定を並べる。
  同じキーがあれば jobs 側が優先される。キー名は main.py の引数名と同じ。

    defaults:
      fps: 30
      date: 20260716
    jobs:
      - data: oneperson_sim_9Light_2x2_01          # 6x6 グリッド: real と混ぜられる形式
      - data: oneperson_sim_9Light_2x2_randomsensor_01
        max_coord: 2.0        # 6x6 グリッド以外は max_coord が必要（従来形式）

  真偽値は --key / --no-key に変換される。それ以外は --key value に変換される。

  ※ batch/ は .gitignore 対象。設定ファイルは各自のローカルに置く。
  詳細は knowledge/batch_run.md を参照。
"""
import os
import sys
import subprocess
import yaml

MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")


def load_jobs(yaml_path: str):
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        print(f"[error] YAML のトップレベルは defaults / jobs を持つマッピングにしてください: {yaml_path}")
        sys.exit(1)

    defaults = cfg.get("defaults") or {}
    jobs = cfg.get("jobs")
    if not jobs:
        print(f"[error] jobs が空です: {yaml_path}")
        sys.exit(1)

    merged = []
    for i, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            print(f"[error] {i} 番目の job がマッピングではありません: {job}")
            sys.exit(1)
        m = {**defaults, **job}
        if "data" not in m:
            print(f"[error] {i} 番目の job に data がありません: {job}")
            sys.exit(1)
        merged.append(m)
    return merged


def build_args(job: dict) -> list:
    """job の各キーを main.py の CLI 引数へ変換する。"""
    args = []
    for key, value in job.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            args.append(flag if value else f"--no-{key}")
        else:
            args += [flag, str(value)]
    return args


def run_job(job: dict, index: int, total: int):
    cmd = [sys.executable, MAIN] + build_args(job)
    print(f"\n{'#'*60}")
    print(f"  [{index}/{total}] {job['data']}")
    print(f"{'#'*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[error] {job['data']} が失敗しました。バッチを中断します。")
        sys.exit(result.returncode)


def main():
    if len(sys.argv) < 2:
        print("使い方: python main_batch.py <設定YAMLのパス>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    if not os.path.exists(yaml_path):
        print(f"[error] ファイルが見つかりません: {yaml_path}")
        sys.exit(1)

    jobs = load_jobs(yaml_path)
    total = len(jobs)
    print(f"\n{total} 件のデータを順番に処理します。")

    for i, job in enumerate(jobs, start=1):
        run_job(job, i, total)

    print(f"\n{'#'*60}")
    print(f"  全 {total} 件の処理が完了しました。")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
