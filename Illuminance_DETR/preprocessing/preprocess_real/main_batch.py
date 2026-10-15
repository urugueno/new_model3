"""
バッチ実行スクリプト（YAML 設定で複数データをまとめて処理）

使い方:
  python .../main_batch.py .../batch/oneperson_gym.yaml
  python .../main_batch.py .../batch/oneperson_gym.yaml --gpu 1   # GPU を指定

  バックグラウンド実行（GPU 指定は --gpu を使う）:
    nohup python .../main_batch.py .../batch/oneperson_gym.yaml --gpu 1 > log_gym1.out 2>&1 &

  注意: `nohup CUDA_VISIBLE_DEVICES=1 python ...` は動かない。nohup は最初の引数を
  実行ファイル名として扱うため、変数代入の前置きは nohup より前に置く必要がある
  （`CUDA_VISIBLE_DEVICES=1 nohup python ...`）。--gpu ならその落とし穴がない。

設定ファイル（YAML）の書き方:
  defaults に共通設定、jobs にデータごとの設定を並べる。
  同じキーがあれば jobs 側が優先される。キー名は main.py の引数名と同じ。

    defaults:
      area_size: 2.0          # 必須（人が動き回る範囲の大きさ・メートル）
      people: 1
    jobs:
      - data: 1_tsunoda_1
        start_img: /nas/.../color/20260617_061634_533468.jpg
      - data: 1_tsunoda_2
        start_img: /nas/.../color/20260617_065232_781104.jpg
        detect: false         # このジョブだけ detect をスキップ（--no-detect）
        z_offset: 0.95        # 手動指定したい場合

  真偽値は --key / --no-key に変換される（例: detect: false -> --no-detect）。
  list / dict の値は JSON 文字列に変換して1引数として渡す（例: persons）。
  それ以外は --key value に変換される。

  複数人（people >= 2）のラベル付けは persons を指定する。人物ごとに、その人物に
  属する track_id 一覧（ids）と開始フェーズ（start_phase, 0〜cycle_phases-1）を並べる。
  track_id は不安定（同一人物でも別番号があり得る）なので、同一人物の id をまとめて記録する:

    jobs:
      - data: twoperson_gym_01
        start_img: /nas/.../color/xxxxx.jpg
        people: 2
        persons:
          - ids: [1, 5, 9]    # この track_id 群 = 人物0
            start_phase: 3     # t=0 で crouch から開始
          - ids: [2, 7]       # 人物1
            start_phase: 0     # t=0 で walk から開始

  track_id は録画の途中で別人に乗り移ることがある（ByteTrack の ID スイッチ）。
  その場合は ids の要素を辞書にして、その id が誰のものである区間かを指定する:

        persons:
          - ids: [1, {id: 9, to: 20260129_083945_703176.jpg}]     # 9 はこの時刻まで人物0
            start_phase: 3
          - ids: [2, {id: 9, from: 20260129_083945_703176.jpg}]   # 以降は人物1
            start_phase: 0

  from は含み to は含まない。同じ時刻を前の区間の to と次の区間の from に書けば
  穴も重なりも出ない。時刻は上のカラー画像ファイル名でも
  "2026-01-29 08:39:45 703176"（timestamp 列と同じ形式）でも書ける。

  同じ id について from だけ / to だけを並べて書いた場合は、書いた順に区間を開いて
  閉じると読む。乗り移りが一時的で元の人物に戻る場合はこう書ける:

        persons:
          - ids: [7, {id: 29, from: A}, {id: 29, to: B}]    # 29 は A〜B だけ人物0
            start_phase: 2
          - ids: [10, {id: 29, to: A}, {id: 29, from: B}]   # A まで と B から は人物1
            start_phase: 3

  収録中にサイクルを 1 回飛ばしてしまい、その時刻以降の位相がずれた場合は shifts で
  補正する。from はその補正が効き始める時刻（含む）、delta はずらすフェーズ数
  （省略すると 1 = 1 回飛ばした）。時刻の書き方は ids の from / to と同じ。

    jobs:
      - data: threeperson_gym_02
        start_img: /nas/.../color/xxxxx.jpg
        persons:
          - ids: [1, 5, 9]
            start_phase: 3
            shifts:                                      # 人物0 だけがずれた場合
              - from: 20260129_084644_593751.jpg
                delta: 1
          - ids: [2, 7]
            start_phase: 0
        shifts:                                          # 全員がずれた場合（job 直下）
          - from: 20260129_084644_593751.jpg

  job 直下の shifts は全行に効き、persons の中の shifts はその人物だけに効く。
  1人データ（persons なし）のずれは job 直下の shifts で直す。両方書けば足し合わさる。
  2 回飛ばしたら要素を 2 つ並べる（効果は累積する）。

  行動サイクル（1フェーズ5秒、phase_sec で変更可・小数可）は cycle_phases / drop_phases / phase_sec で指定する。

    defaults:
      cycle_phases: 6     # meeting: walk,walk,stop,crouch,banzai,bow
      drop_phases: [5]    # お辞儀のフレームを破棄（誰か1人でも該当すればフレームごと）

  未指定なら cycle_phases=5（gym: walk,walk,stop,crouch,banzai）で従来どおり動く。
  これらを変更したら relabel だけでなくパイプライン全体を回し直すこと
  （除外はステップ [7] preprocess_real にあり、pose.csv の座標も変わるため）。

  センサが埋まったマットそのものをずらして収録した場合は、その変位を sensor_shift /
  sensor_rot で与える（座標の定義は sensor_layout.py、確認図は check/sensor_layout.png・
  sensor_shadow.png）。向きはカメラから見て +x=右・+y=奥、回転は上から見て反時計回りが正。

  ずらした配置は既定の正規化枠（legacy: グリッド (0,0) が 0・2 m が 1）からはみ出すので、
  sensor_frame: special を併せて指定する。

    defaults:
      sensor_frame: special
    jobs:
      - data: shift_03
        sensor_shift: [0.4, -0.2]              # 右に 40 cm・手前に 20 cm
      - data: rotate
        sensor_rot: {deg: 15, pivot: [5, 5]}   # 右奥のセンサ（グリッド 5,5）を支点に

  隣接関節間距離（骨長）による関節フィルタは既定で有効。人物ごとの骨長の中央値から
  外れた関節を NaN 化し、depth を取り違えて背景に飛んだ関節が教師データに残るのを防ぐ。
  平滑化の前に掛かるので、短い欠損は補間で復元される。

    defaults:
      joint_dist: false     # フィルタを無効化（--no-joint_dist。比較実験用）
      joint_dist_tol: 0.25  # 相対許容幅を厳しくする（既定 0.35）

  tol を変える前に、実行ログの "Reference bone lengths" に出る基準長を確認すること
  （前腕 ≈ 0.25 m、肩幅 ≈ 0.35-0.40 m あたりが妥当）。
"""
import os
import sys
import json
import argparse
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
        if value is None:
            # YAML に `key:` と値なしで書かれた場合。文字列 "None" を渡すと
            # 受け取り側（json.loads など）が落ちるので、未指定として扱う。
            continue
        if isinstance(value, bool):
            args.append(flag if value else f"--no-{key}")
        elif isinstance(value, (list, dict)):
            # persons のようなネスト構造は JSON 文字列で1引数として渡す
            # （subprocess.run はシェルを介さないため argv 1要素として安全）
            # default=str: YAML が日付として解釈した値（to: 2026-01-29 のような
            # 引用符なしの書き方）でも落とさず文字列にして渡し、受け取り側で
            # 「時刻として解釈できません」と案内させる。
            args += [flag, json.dumps(value, ensure_ascii=False, default=str)]
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
    parser = argparse.ArgumentParser(
        description="YAML 設定で複数データをまとめて前処理する",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("yaml_path", help="設定 YAML のパス")
    parser.add_argument("--gpu", default=None, metavar="ID",
                        help="使用する GPU を指定する（CUDA_VISIBLE_DEVICES に設定）。\n"
                             "例: --gpu 0 / --gpu 1 / --gpu 0,1\n"
                             "GPU を使うのは detect_yolo だけなので detect: false 時は影響しない。\n"
                             "YAML 側に gpu: を書いた場合はそちらが優先される（ジョブ単位の指定）")
    args = parser.parse_args()

    # 子プロセス（main.py -> tools/*.py）は os.environ を継承する。
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    yaml_path = args.yaml_path
    if not os.path.exists(yaml_path):
        print(f"[error] ファイルが見つかりません: {yaml_path}")
        sys.exit(1)

    jobs = load_jobs(yaml_path)
    total = len(jobs)
    print(f"\n{total} 件のデータを順番に処理します。")
    if args.gpu is not None:
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    for i, job in enumerate(jobs, start=1):
        run_job(job, i, total)

    print(f"\n{'#'*60}")
    print(f"  全 {total} 件の処理が完了しました。")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
