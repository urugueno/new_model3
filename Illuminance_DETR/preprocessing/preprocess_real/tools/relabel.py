"""
bbox_label.csv に行動ラベルを付与し bbox_relabel.csv を生成する。
その後ファイルをリネームして最終成果物を bbox_label.csv にまとめる。

前提:
- 1フェーズ --phase_sec 秒（既定 5 秒）。1サイクルのフェーズ数は --cycle_phases で与える
  （gym は 5、meeting は お辞儀 を含む 6）。phase → label は 4 クラス。
- 位相の時刻原点は --start_img（しゃがみ始め）の時刻。生き残った最初の行を
  原点にすると、前段のフィルタで先頭フレームが落ちたときに全ラベルがずれる。
- --drop_phases で除外したフェーズが残っていないかを検証する（ステップ [7]
  preprocess_real.py と食い違っていないことの相互チェック）。
- 収録中にサイクルを飛ばしてそれ以降の位相がずれた場合は --shifts で補正する
  （[{"from": <時刻>, "delta": 1}]）。人物ごとのずれは --persons の各要素に
  shifts を書く。[7] と同じ値を渡すこと。

ラベル付けの2モード:
- 1人（--persons 未指定）: person_id は bbox_label.csv から自動取得し、
  開始フェーズはしゃがみ（phase=3）固定。従来どおりの挙動。
- 複数人（--persons 指定）: 人物ごとに track_id 一覧と開始フェーズを与える。
  track_id は不安定（同一人物でも別番号があり得る）ため、
  track_id -> 人物 -> その人物の start_phase を解決してラベルを付ける。
  --persons は JSON 文字列で、[{"ids": [...], "start_phase": int}, ...] 形式。

フェーズ計算そのものは phase.py に集約している（preprocess_real.py と共有）。

ラベル付与後・CSV 書き出し前に、同時刻のラベル個数チェック（verify_labels.py）をかける。
全員が別フェーズから始まる前提なので、1 フレームに同じ行動は並ばない。ただし walk だけは
phase 0/1 の 2 フェーズぶんあるので 2 つまで許す。違反したら CSV を書かずに止め、
該当フレーム前後のクリップを check/violation_NN.mp4 に出す。
"""
import os
import sys
import shutil
import argparse
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import processed_dir
from phase import (DEFAULT_CYCLE_PHASES, PHASE_DURATION_SEC, TIME_FORMAT, anchor_timestamp, compute_phase,
                   parse_drop_phases_arg, parse_persons_arg, parse_shifts_arg,
                   phase_to_label, resolve_start_phase, validate_cycle)


def run(data_name: str, persons: list = None, start_img: str = None,
        cycle_phases: int = DEFAULT_CYCLE_PHASES, drop_phases: list = (),
        area_size: float = 2.0, verify: bool = True, shifts=(),
        phase_sec: float = PHASE_DURATION_SEC):
    data_dir = processed_dir(data_name)
    bbox_path = os.path.join(data_dir, "bbox_label.csv")
    relabel_path = os.path.join(data_dir, "bbox_relabel.csv")
    before_path = os.path.join(data_dir, "bbox_beforelabel.csv")

    if not os.path.exists(bbox_path):
        print(f"[relabel] [error] bbox_label.csv が見つかりません: {bbox_path}")
        sys.exit(1)

    df = pd.read_csv(bbox_path)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format=TIME_FORMAT)
    df = df.sort_values("timestamp_dt")

    # preprocess_real.py は常に label=0 で書き出すので、0 以外があれば付与済みのファイル。
    # relabel.py だけを 2 回流すと bbox_beforelabel.csv が付与済みで上書きされ、
    # さらに drop_phases の除外（ステップ [7]）が効かないままになる。
    if "label" in df.columns and (df["label"].fillna(0) != 0).any():
        print(f"[relabel] [warn] 既にラベル付け済みに見えます（label に 0 以外があります）")
        print(f"[relabel] [warn] relabel 単体ではなく main.py / main_batch.py で回し直してください")

    df["start_phase"] = resolve_start_phase(df, persons, cycle_phases, "[relabel]",
                                            strict=True, shifts=shifts)

    phase = compute_phase(df["timestamp_dt"], anchor_timestamp(start_img),
                          df["start_phase"], cycle_phases, "[relabel]", phase_sec)

    # 相互チェック: ステップ [7] で除外したはずのフェーズが残っていないか。
    # [8] は [7] の行の部分集合で原点も計算経路も同一なので、本来ここは発火しない。
    if drop_phases:
        leaked = phase.isin(drop_phases)
        if leaked.any():
            sample = df.loc[leaked, "timestamp"].head(3).tolist()
            print(f"[relabel] [error] 除外したはずのフェーズ {list(drop_phases)} の行が "
                  f"{int(leaked.sum())} 件残っています: {sample}")
            print(f"[relabel]         preprocess_real.py と引数が食い違っているか、"
                  f"古い bbox_label.csv を読んでいます。パイプライン全体を回し直してください")
            sys.exit(1)

    df["label"] = phase_to_label(phase, "[relabel]")

    import verify_labels

    # 前回実行の違反ファイルは、今回の検査結果にかかわらず必ず消す。
    # --no-verify や 1人モードでも残骸が残らないよう、チェックの外で消す。
    removed = verify_labels.clear_violation_files(os.path.join(data_dir, "check"))
    if removed:
        print(f"[relabel] 前回の violation ファイル {removed} 件を削除しました")

    # 同時刻のラベル個数チェック（walk は 2 つまで、他は 1 つまで）。
    # ids の割り当てミスはここで初めて観測可能になるので、CSV を書き出す前に止める。
    # 1人モードは 1 フレーム 1 行なので素通りするが、走らせて困るものでもない。
    if verify:
        n_bad = verify_labels.check(data_name, df, area_size=area_size)
        if n_bad:
            print(f"[relabel] [error] ラベル個数チェックに失敗したため中断します"
                  f"（bbox_label.csv は書き換えていません）")
            sys.exit(1)

    df = df.drop(columns=["timestamp_dt", "start_phase"])
    df.to_csv(relabel_path, index=False)
    print(f"[relabel] bbox_relabel.csv を生成しました")

    # リネーム
    shutil.move(bbox_path, before_path)
    shutil.move(relabel_path, bbox_path)
    print(f"[relabel] bbox_label.csv → bbox_beforelabel.csv")
    print(f"[relabel] bbox_relabel.csv → bbox_label.csv")
    print("[relabel] 完了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--start_img", required=True,
                        help="しゃがみ始めのカラー画像。その時刻が位相の原点になる")
    parser.add_argument("--persons", default=None,
                        help='複数人用: [{"ids": [...], "start_phase": int}, ...] 形式の JSON 文字列')
    parser.add_argument("--cycle_phases", type=int, default=DEFAULT_CYCLE_PHASES,
                        help=f"1サイクルのフェーズ数（既定: {DEFAULT_CYCLE_PHASES}）。"
                             "gym は 5、meeting は お辞儀 を含む 6")
    parser.add_argument("--phase_sec", type=float, default=PHASE_DURATION_SEC,
                        help=f"1フェーズの秒数（既定: {PHASE_DURATION_SEC}）。小数可。"
                             "preprocess_real.py と同じ値を渡すこと")
    parser.add_argument("--drop_phases", default=None,
                        help="除外済みフェーズの JSON 配列 例: '[5]'。残っていたらエラーにする")
    parser.add_argument("--shifts", default=None,
                        help='全行に効く位相補正の JSON 配列 '
                             '例: \'[{"from": "20260129_084644_593751.jpg", "delta": 1}]\'。'
                             'その時刻以降の位相を delta だけ進める（delta 省略時は 1）。'
                             '収録中にサイクルを飛ばしたときの補正で、'
                             '人物ごとのずれは --persons の各要素に shifts を書く')
    parser.add_argument("--area_size", type=float, default=2.0,
                        help="違反クリップの 3D パネル軸範囲（前処理の --area_size と同じ値）")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                        help="同時刻のラベル個数チェック（walk は 2 つまで、他は 1 つまで）を"
                             "行うか（既定: する）")
    args = parser.parse_args()

    persons = parse_persons_arg(args.persons, "[relabel]")
    drop_phases = parse_drop_phases_arg(args.drop_phases, "[relabel]")
    shifts = parse_shifts_arg(args.shifts, "[relabel]")
    validate_cycle(args.cycle_phases, drop_phases, "[relabel]", args.phase_sec)

    run(args.data, persons=persons, start_img=args.start_img,
        cycle_phases=args.cycle_phases, drop_phases=drop_phases,
        area_size=args.area_size, verify=args.verify, shifts=shifts,
        phase_sec=args.phase_sec)
