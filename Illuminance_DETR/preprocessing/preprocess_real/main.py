"""
データ加工パイプライン エントリポイント

使い方:
  python pipeline.py --data ueno_1 --start_img 20260520_064853_041393.jpg --people 1

  # shift/offset を手動指定する場合:
  python pipeline.py --data ueno_1 --start_img 20260520_064853_041393.jpg --people 1 \
      --shift -000000.026351 \
      --x_offset 1.5 --y_offset 1.25 --z_offset 0.812

入力  : <DATA_ROOT>/raw/light/<データ名>/     （読み取りのみ・書き換えない）
出力  : <DATA_ROOT>/processed/light/<データ名>/

pose_origin.csv は YOLO の生の推定値（未検出の関節は空欄のまま）で保存し、
欠損の補間と平滑化（中央値→移動平均）は [7] preprocess_real 側でまとめて行う。
窓幅は --smooth_median_window / --smooth_mean_window / --smooth_max_gap で調整でき、
--no-smooth で平滑化なしの比較もできる。
--smooth_max_gap の既定は 0.38 秒（33.34ms/frame で約 10 フレームの欠損まで補間する）。
これを超える欠損は埋めず、関節フィルタで行ごと落とす。

行動サイクルは 1 フェーズ 5 秒（--phase_sec で変更可、小数可）。gym は 5 フェーズ（walk,walk,stop,crouch,hand）、
meeting は末尾に お辞儀 が入る 6 フェーズ。お辞儀は実験プロトコル外の動作なので、
--cycle_phases 6 --drop_phases '[5]' でフレームごと除外する（複数人データでは
誰か1人でもお辞儀していればそのフレーム全体を落とす）。位相の時刻原点は常に
--start_img の時刻で、[7] と [8] に同じ値が渡る。

収録中にサイクルを 1 回飛ばすと、その時刻以降の位相が丸ごとずれる。原点と start_phase は
録画を通して一定なので、これは --shifts で補正する（その時刻以降の位相を delta だけ進める）。

  # 08:46:44 以降、全員の位相が 1 つ進んでいる場合:
  python main.py --data threeperson_gym_02 --start_img ... --area_size 2.0 \
      --shifts '[{"from": "20260129_084644_593751.jpg", "delta": 1}]'

ずれたのが特定の人物だけなら --persons のその人物に shifts を書く（両方書けば足し合わさる）。

  # meeting（6フェーズ・お辞儀除外）:
  python main.py --data oneperson_meeting_01 --start_img ... --people 1 --area_size 2.0 \
      --cycle_phases 6 --drop_phases '[5]'

[8] relabel でラベル付けした直後に、同時刻のラベル個数チェックを行う。全員が別フェーズ
から始まる前提なので、1 フレームに同じ行動は並ばない（stop/crouch/hand は各 1 つまで）。
walk だけは phase 0/1 の 2 フェーズぶんあるため 2 つまで許す。違反したら CSV を書かずに止め、
該当フレーム前後 3 秒のクリップを check/violation_NN.mp4 に出す。bbox の上には
ラベルではなく person_id を出すので、どの id の割り当てが誤っているか目で追える。

センサが埋まったマットそのものをずらして収録した場合は、その変位を --sensor_shift /
--sensor_rot で与える（カメラから見て +x=右・+y=奥、上から見て反時計回りが正）。
[7.5] sensor_coords.py が sensor_coords.json（センサ ID -> 正規化座標）を書き、
check/ に sensor_layout.png・sensor_shadow.png を出す。座標の定義は sensor_layout.py。

正規化枠は既定で legacy（グリッド (0,0) が 0・2 m が 1）。ずらした配置は legacy からはみ出すので、
special 収録は --sensor_frame special（[-0.6, 2.6] m を [0, 1]）で処理する。

  # マットが右に 40 cm・手前に 20 cm ずれた:
  python main.py --data shift_03 ... --sensor_frame special --sensor_shift '[0.4, -0.2]'
  # 右奥のセンサ（グリッド 5,5）を支点に反時計回り 15°:
  python main.py --data rotate ... --sensor_frame special --sensor_rot '{"deg": 15, "pivot": [5, 5]}'

実行後に確認:
  processed/light/<データ名>/check/ → シフト検出結果（shift.txt・CSV）・
                                      bbox 座標グラフ・採用オフセット値・
                                      照度/RGB/pose/bbox を並べた目視検証クリップ・
                                      センサ配置図（sensor_layout.png / sensor_shadow.png）
"""
import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import processed_dir

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")


def run_script(script: str, extra_args: list = None):
    cmd = [sys.executable, os.path.join(TOOLS, script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"  実行: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[error] {script} が失敗しました（終了コード: {result.returncode}）")
        sys.exit(result.returncode)


def run_script_optional(script: str, extra_args: list = None):
    """
    検証用の付加物を実行する。本体成果物は既に生成済みなので、
    ここで失敗してもパイプライン全体は落とさず警告に留める。
    """
    cmd = [sys.executable, os.path.join(TOOLS, script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"  実行: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[warn] {script} が失敗しました（終了コード: {result.returncode}）")
        print(f"[warn] 検証用動画の生成のみ失敗しています。本体の成果物は生成済みです。")
    return result.returncode == 0


def read_shift_from_file(data_name: str) -> str:
    shift_path = os.path.join(processed_dir(data_name), "check", "shift.txt")
    with open(shift_path) as f:
        return f.read().strip()


def check_start_img(data_name: str, start_img: str):
    """
    --start_img の時刻が timestamp.csv に含まれているか確認する。

    後段は start_img を時刻のしきい値として使うだけで、存在確認をしない。color/ には
    timestamp.csv に載っていない別テイクの画像が混ざっていることがあり、それを指定しても
    trim が何も削らず、位相の原点もずれたまま黙って最後まで走る（sound_12 で実際に起きた）。
    """
    import pandas as pd
    from phase import filename_to_timestamp

    try:
        start_ts = filename_to_timestamp(start_img)
    except ValueError as e:
        print(f"[error] {e}")
        sys.exit(1)

    timestamps = pd.read_csv(os.path.join(processed_dir(data_name), "timestamp.csv"))["timestamp"]
    if (timestamps == start_ts).any():
        print(f"[ok] start_img の時刻 {start_ts} は timestamp.csv に含まれています")
        return

    timestamps = timestamps.sort_values()
    print(f"[error] start_img の時刻 {start_ts} が timestamp.csv に含まれていません: {start_img}")
    print(f"        timestamp.csv の範囲: {timestamps.iloc[0]} 〜 {timestamps.iloc[-1]}")
    before = timestamps[timestamps < start_ts]
    after = timestamps[timestamps > start_ts]
    if len(before):
        print(f"        直前のフレーム: {before.iloc[-1]}")
    if len(after):
        print(f"        直後のフレーム: {after.iloc[0]}")
    sys.exit(1)


def read_z_offset_from_file(data_name: str):
    values_path = os.path.join(processed_dir(data_name), "check", "offset_values.txt")
    offsets = {}
    with open(values_path) as f:
        for line in f:
            k, v = line.strip().split("=")
            offsets[k] = v
    return offsets["z_offset"]


def main():
    parser = argparse.ArgumentParser(
        description="データ加工パイプライン",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data", required=True, help="データフォルダ名 例: ueno_1")
    parser.add_argument("--start_img", required=True,
                        help="しゃがみ始めのカラー画像ファイル名 例: 20260520_064853_041393.jpg")
    parser.add_argument("--people", type=int, default=1, help="人数（デフォルト: 1）")
    parser.add_argument("--area_size", type=float, required=True,
                        help="人が動き回れる床面の大きさ（メートル）。pose をこのサイズにリスケールし中央原点で出力")
    parser.add_argument("--rescale_ref", type=str, default=None, choices=["ankle", "hip"],
                        help="--area_size に合わせる基準。ankle（既定）は足首の広がり＝実際に踏んだ床面。"
                             "hip は腰の軌跡を合わせる旧挙動で、X/Y が約 1.13 倍に過大化するため"
                             "修正前の結果を再現するとき以外は使わない")
    parser.add_argument("--bbox_size", type=float, default=None,
                        help="bbox の水平（X/Y）一辺の長さ（メートル）。hip 中心。"
                             "未指定なら preprocess_real.py の既定値 BBOX_SIZE_M（0.5 m）")
    parser.add_argument("--shift", default=None, help="時刻ずれを手動指定 例: -000000.026351")
    parser.add_argument("--z_offset", type=float, default=None, help="z_offset（床基準）手動指定")
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=True,
                        help="pose の時系列平滑化（中央値→移動平均）を行うか（デフォルト: する）。"
                             "--no-smooth で pose_origin の生値をそのまま使う")
    parser.add_argument("--smooth_median_window", type=int, default=None,
                        help="スパイク除去用 rolling median の窓（フレーム数）。未指定なら preprocess_real.py の既定値")
    parser.add_argument("--smooth_mean_window", type=int, default=None,
                        help="平滑化用 rolling mean の窓（フレーム数）。未指定なら preprocess_real.py の既定値")
    parser.add_argument("--smooth_max_gap", type=float, default=None,
                        help="補間する欠損の最大長（秒）。未指定なら preprocess_real.py の既定値 0.38 秒"
                             "（約 10 フレーム）。これを超える欠損は埋めない")
    parser.add_argument("--joint_dist", action=argparse.BooleanOptionalAction, default=True,
                        help="隣接関節間距離（骨長）による関節フィルタを行うか（デフォルト: する）。"
                             "人物ごとの骨長の中央値から外れた関節を NaN 化し、depth を取り違えて"
                             "背景に飛んだ関節が教師データに残るのを防ぐ。"
                             "平滑化の前に掛けるので、短い欠損は補間で復元される。"
                             "--no-joint_dist で無効化")
    parser.add_argument("--joint_dist_tol", type=float, default=None,
                        help="骨長フィルタの相対許容幅（長すぎ側）。"
                             "未指定なら preprocess_real.py の既定値 0.8")
    parser.add_argument("--joint_dist_short_tol", type=float, default=None,
                        help="骨長フィルタの相対許容幅（短すぎ側）。既定は無効。"
                             "単眼 depth では視線方向を向いた骨の距離が原理的に潰れ、"
                             "その向きは動作と相関するため、有効にすると特定の動作クラスだけが"
                             "大量に消える（実測: crouch フェーズの 97.9%%）。"
                             "使う場合はフェーズ別の除去率を必ず確認すること")
    parser.add_argument("--detect", action=argparse.BooleanOptionalAction, default=True,
                        help="YOLO 3D ポーズ推定を実行するか（デフォルト: する）。"
                             "--no-detect で既存の pose_origin.csv を再利用しスキップ。"
                             "注意: 2026-09-11 以前に作られた pose_origin.csv は移動平均済みなので、"
                             "生値から作り直すには --detect で再実行すること")
    parser.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "sam3"],
                        help="[3] detect_yolo の person_id の付け方。sam3 は ID が途切れにくく "
                             "persons の ids が 1 人 1 つで済むことが多いが、約 7 fps と遅い")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True,
                        help="目視検証用クリップ（check/）を生成するか（デフォルト: する）。"
                             "--no-check でスキップ")
    parser.add_argument("--check_clips", type=int, default=5,
                        help="検証クリップの本数（デフォルト: 5）。"
                             "先頭 1 本と末尾 1 本は必ず含み、残りをランダムに選ぶ")
    parser.add_argument("--check_duration", type=float, default=10.0,
                        help="検証クリップ1本の長さ・秒（デフォルト: 10）")
    parser.add_argument("--check_seed", type=int, default=0,
                        help="検証クリップの抽出タイミングの乱数シード（デフォルト: 0）")
    parser.add_argument("--persons", default=None,
                        help='複数人ラベル付け用の JSON 文字列。'
                             '[{"ids": [...], "start_phase": int}, ...] 形式で '
                             'preprocess_real.py / relabel.py に転送する。'
                             'ids の要素は track_id そのもの（全区間）か、'
                             '{"id": 1, "from": ..., "to": ...}（その区間だけ）。'
                             'track_id が途中で別人に乗り移る場合は区間で分ける。'
                             '未指定なら1人モード（従来動作）')
    parser.add_argument("--shifts", default=None,
                        help='収録中にサイクルを飛ばしたときの位相補正。'
                             '[{"from": <時刻>, "delta": 1}, ...] 形式の JSON 文字列で、'
                             'その時刻以降の位相を delta だけ進める（delta 省略時は 1）。'
                             '時刻はカラー画像のファイル名でも timestamp 文字列でも可。'
                             'preprocess_real.py / relabel.py の両方に同じ値を転送する。'
                             '人物ごとのずれは --persons の各要素に shifts を書く')
    parser.add_argument("--cycle_phases", type=int, default=5,
                        help="1サイクルのフェーズ数（デフォルト: 5）。"
                             "gym は 5（walk,walk,stop,crouch,hand）、"
                             "meeting は お辞儀 を含む 6")
    parser.add_argument("--phase_sec", type=float, default=5.0,
                        help="1フェーズの秒数（デフォルト: 5.0）。小数可（例: 2.5）。"
                             "[7] preprocess_real と [8] relabel に同じ値を転送する")
    parser.add_argument("--drop_phases", default=None,
                        help='フレームごと除外するフェーズの JSON 配列 例: \'[5]\'（meeting のお辞儀）。'
                             '複数人データでは誰か1人でも該当すればそのフレームを落とす')
    parser.add_argument("--sensor_shift", default=None, metavar="JSON",
                        help="センサマットの平行移動 '[dx, dy]'（メートル）。カメラから見て +x=右・+y=奥。"
                             "例: 右に 40 cm・手前に 20 cm -> '[0.4, -0.2]'。未指定なら変位なし")
    parser.add_argument("--sensor_rot", default=None, metavar="JSON",
                        help='センサマットの回転 \'{"deg": 15, "pivot": [5, 5]}\'。上から見て反時計回りが正、'
                             "pivot は動かなかったセンサのグリッド番号（左手前 0,0・右奥 5,5）。未指定なら回転なし")
    parser.add_argument("--sensor_frame", default=None, choices=["legacy", "special"],
                        help="センサ座標の正規化枠。未指定なら legacy（グリッド (0,0) が 0・2 m が 1、従来どおり）。"
                             "special は [-0.6, 2.6] m を [0, 1] にする枠で、2026-10-09 の special 収録用"
                             "（マットをずらしても [0, 1] に収まる）。sensor_layout.py 参照")
    parser.add_argument("--gpu", default=None, metavar="ID",
                        help="使用する GPU を指定する（CUDA_VISIBLE_DEVICES に設定）。"
                             "例: --gpu 0 / --gpu 1 / --gpu 0,1。"
                             "GPU を使うのは [3] detect_yolo だけなので --no-detect 時は影響しない")
    args = parser.parse_args()

    # 子プロセス（tools/*.py）は os.environ を継承するので、ここで設定すれば全ステップに効く。
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"\n{'#'*60}")
    print(f"  パイプライン開始: {args.data}")
    if args.gpu is not None:
        print(f"  CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"{'#'*60}")

    # [1] setup: パス修正・ヘッダー変換
    run_script("setup_data.py", ["--data", args.data])

    # [2] duplicate_check: タイムスタンプ重複確認
    run_script("duplicate_check.py", ["--data", args.data])

    # start_img が timestamp.csv の行と一致するか確認（範囲外の別テイク画像の指定を止める）
    check_start_img(args.data, args.start_img)

    # [3] detect_yolo: 3D ポーズ推定（--no-detect でスキップ）
    if args.detect:
        detect_args = ["--data", args.data, "--tracker", args.tracker]
        if args.tracker == "sam3":
            # start_img 以前は trim_pose で消えるので処理しない（全員がそろった時点から ID を振る）
            detect_args += ["--start_img", args.start_img]
        run_script("detect_yolo.py", detect_args)
    else:
        pose_origin_path = os.path.join(processed_dir(args.data), "pose_origin.csv")
        if not os.path.exists(pose_origin_path):
            print(f"[error] --no-detect ですが pose_origin.csv が見つかりません: {pose_origin_path}")
            sys.exit(1)
        print(f"  → detect をスキップし既存の pose_origin.csv を使用します")

    # [4] trim_pose: スタート位置以前を削除
    run_script("trim_pose.py", ["--data", args.data, "--start_img", args.start_img])

    # [5] calc_shift: 時刻同期
    shift_args = ["--data", args.data]
    if args.shift:
        shift_args += ["--shift", args.shift]
    run_script("calc_shift.py", shift_args)
    shift = args.shift or read_shift_from_file(args.data)
    print(f"  → 使用するシフト値: {shift}")

    # [6] calc_offset: z_offset（床基準）を自動計算
    offset_args = ["--data", args.data]
    if args.z_offset is not None:
        offset_args += ["--z_offset", str(args.z_offset)]
    run_script("calc_offset.py", offset_args)
    z_offset = str(args.z_offset) if args.z_offset is not None else read_z_offset_from_file(args.data)
    print(f"  → 使用する z_offset: {z_offset}")

    # [7] preprocess_real: 前処理（--fix-z 付きで最終実行）
    preprocess_args = [
        "--data", args.data,
        "--shift", shift,
        "--people", str(args.people),
        "--area_size", str(args.area_size),
        "--z_offset", str(z_offset),
        "--fix-z",
        # 位相の時刻原点とサイクル定義（お辞儀フレームの除外に使う）
        "--start_img", args.start_img,
        "--cycle_phases", str(args.cycle_phases),
        "--phase_sec", str(args.phase_sec),
    ]
    if args.rescale_ref is not None:
        preprocess_args += ["--rescale_ref", args.rescale_ref]
    if args.bbox_size is not None:
        preprocess_args += ["--bbox_size", str(args.bbox_size)]
    if args.persons:
        preprocess_args += ["--persons", args.persons]
    if args.drop_phases:
        preprocess_args += ["--drop_phases", args.drop_phases]
    if args.shifts:
        preprocess_args += ["--shifts", args.shifts]
    if not args.smooth:
        preprocess_args += ["--no-smooth"]
    if args.smooth_median_window is not None:
        preprocess_args += ["--smooth_median_window", str(args.smooth_median_window)]
    if args.smooth_mean_window is not None:
        preprocess_args += ["--smooth_mean_window", str(args.smooth_mean_window)]
    if args.smooth_max_gap is not None:
        preprocess_args += ["--smooth_max_gap", str(args.smooth_max_gap)]
    if not args.joint_dist:
        preprocess_args += ["--no-joint_dist"]
    if args.joint_dist_tol is not None:
        preprocess_args += ["--joint_dist_tol", str(args.joint_dist_tol)]
    if args.joint_dist_short_tol is not None:
        preprocess_args += ["--joint_dist_short_tol", str(args.joint_dist_short_tol)]
    run_script("preprocess_real.py", preprocess_args)

    # [7.5] sensor_coords: センサ ID -> 座標（sensor_coords.json）と配置の確認図
    sensor_args = ["--data", args.data]
    if args.sensor_shift:
        sensor_args += ["--sensor_shift", args.sensor_shift]
    if args.sensor_rot:
        sensor_args += ["--sensor_rot", args.sensor_rot]
    if args.sensor_frame:
        sensor_args += ["--sensor_frame", args.sensor_frame]
    if not args.check:
        sensor_args += ["--no-plot"]
    run_script("sensor_coords.py", sensor_args)

    # [8] relabel: ラベル付け・リネーム
    # 原点・サイクル定義は [7] と必ず同じ値を渡す（食い違うと relabel 側でエラーになる）
    relabel_args = [
        "--data", args.data,
        "--start_img", args.start_img,
        "--cycle_phases", str(args.cycle_phases),
        "--phase_sec", str(args.phase_sec),
        # 多重集合チェックに引っかかったときの違反クリップ用（[9] と同じ軸範囲にする）
        "--area_size", str(args.area_size),
    ]
    if args.persons:
        relabel_args += ["--persons", args.persons]
    if args.drop_phases:
        relabel_args += ["--drop_phases", args.drop_phases]
    if args.shifts:
        relabel_args += ["--shifts", args.shifts]
    run_script("relabel.py", relabel_args)

    # [9] make_check_videos: 目視検証用クリップ（--no-check でスキップ）
    if args.check:
        run_script_optional("make_check_videos.py", [
            "--data", args.data,
            "--area_size", str(args.area_size),
            "--clips", str(args.check_clips),
            "--duration", str(args.check_duration),
            "--seed", str(args.check_seed),
        ])
    else:
        print(f"\n  → check をスキップします（--no-check）")

    print(f"\n{'#'*60}")
    print(f"  パイプライン完了: {args.data}")
    out_dir = processed_dir(args.data)
    print(f"  確認ファイル:")
    check_note = "シフト検出結果・bbox 座標グラフ・採用オフセット値・センサ配置図"
    if args.check:
        check_note += "・照度/RGB/pose/bbox の目視検証クリップ"
    print(f"    {out_dir}/check/ → {check_note}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
