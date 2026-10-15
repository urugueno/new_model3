"""
同時刻のラベル多重集合チェック（複数人データの ids 割り当て検証）。

撮影プロトコル上、複数人データでは全員が別々のフェーズから行動サイクルを始める。
したがって同一フレームに同じ行動が並ぶことはない。ただし walk だけは phase 0 と 1 の
2 フェーズ（計 10 秒）を占めるため、2 人が同時に walk になるのは正常。

  判定基準: 1 フレームにつき walk は 2 つまで、stop / crouch / hand は 1 つまで。

start_phase の多重集合と厳密に突き合わせる方法もあるが、それだと 2 人が同じ
start_phase に割り当てられていて両方 walk のフレームまで違反になる。その取り違えは
サイクルが進んで walk 以外のラベルに変わった時刻で必ず引っかかるので、ここでは
ラベルの個数だけを見る（上の基準は人数やサイクル長に依存しないので全データ共通）。

検出できないもの:
  2 人の ids をそっくり入れ替えた場合、ラベルの個数は正しいままなので通過する。
  これはラベルが時刻から計算される以上、姿勢や速度などの物理量と突き合わせない限り
  検出できない（ラベル同士の整合だけでは反証できない）。

check/ の violation を含むファイル（violation_NN.mp4 / label_violations.txt）は、
違反の有無にかかわらず毎回まず削除してから書き出す。残しておくと、前回の残骸なのか
今回の検出結果なのかが名前から判別できず、直したつもりのデータを疑い続けることになる。

違反したときの出力（いずれも check/ 配下）:
- label_violations.txt : 該当フレームのカラー画像名・実ラベル・person_id の一覧。
                         画像名は ids の from / to にそのまま貼れる形式にしてある。
- violation_NN.mp4     : 該当フレームの前後 --margin 秒（既定 3 秒）のクリップ。
                         違反が長く続く場合も 1 本は margin + --body 秒 + margin で
                         打ち切る（違反の序盤だけあれば原因は追える）。
                         レイアウトは make_check_videos.py と同じ 2x2 で、bbox の上には
                         ラベルではなく person_id（YAML の ids）を出す。違反フレームは
                         赤枠で強調するので、どの id が誤って割り当てられたか目で追える。

使い方（単体実行。ラベル付け済みの bbox_label.csv を検査する。
ラベルの個数しか見ないので --persons は不要）:
  python verify_labels.py --data threeperson_gym_01 --area_size 2.0
"""
import os
import sys
import argparse
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import processed_dir
from phase import TIME_FORMAT, timestamp_to_filename

DEFAULT_MARGIN_SEC = 3.0    # 違反フレームの前後に取る秒数
DEFAULT_MAX_CLIPS = 5       # 生成するクリップの上限（違反が連続すると際限なく増えるため）
DEFAULT_BODY_SEC = 3.0      # 違反区間から切り出す長さ（序盤だけ見れば原因は分かる）
LABEL_NAMES = {0: "walk", 1: "stop", 2: "crouch", 3: "hand"}
# 1 フレームに同時に存在してよい個数。walk だけ phase 0/1 の 2 フェーズ分あるので 2。
MAX_PER_LABEL = {0: 2, 1: 1, 2: 1, 3: 1}


def _labels_text(labels) -> str:
    """その時刻に実際に付いたラベルを名前で並べる（レポート表示用）。"""
    if not labels:
        return "-"
    return ",".join(LABEL_NAMES.get(v, "?") for v in labels)


def clear_violation_files(check_dir: str) -> int:
    """
    前回実行で残った violation 系ファイルを check/ から消し、削除数を返す。

    違反が無くなったときに古い violation_01.mp4 が残っていると、
    「直ったのか、今回の検出結果なのか」が分からなくなるため、
    検査結果にかかわらず毎回先に消す。
    """
    if not os.path.isdir(check_dir):
        return 0
    removed = 0
    for name in sorted(os.listdir(check_dir)):
        path = os.path.join(check_dir, name)
        if "violation" in name and os.path.isfile(path):
            os.remove(path)
            removed += 1
    return removed


def _frame_name(group: pd.DataFrame) -> str:
    """
    違反フレームをカラー画像のファイル名で示す。
    YAML の ids の from / to にそのまま貼れる形式にしておく。
    image_path があればそれを使い（実ファイル名そのもの）、無ければ timestamp から作る。
    """
    if "image_path" in group.columns:
        path = group["image_path"].iloc[0]
        if isinstance(path, str) and path:
            return os.path.basename(path)
    return timestamp_to_filename(group["timestamp"].iloc[0])


def find_violations(df: pd.DataFrame) -> list:
    """
    タイムスタンプごとにラベルの個数を数え、上限を超えたフレームを返す。

    df には label 列（ラベル付与後）と person_id 列が必要。
    label が欠損している行は数えられないので、そのフレームは違反として報告する。
    """
    violations = []
    for ts, group in df.groupby("timestamp", sort=False):
        labels = sorted(int(v) for v in group["label"] if pd.notna(v))
        if len(labels) != len(group):
            exceeded = ["(label 欠損)"]
        else:
            counts = Counter(labels)
            exceeded = [LABEL_NAMES.get(v, str(v)) for v, n in sorted(counts.items())
                        if n > MAX_PER_LABEL.get(v, 1)]
        if exceeded:
            violations.append({
                "timestamp": ts,
                # 動画のフレーム照合には timestamp を、人が読む欄には画像名を使う
                "image": _frame_name(group),
                "labels": labels,
                "exceeded": exceeded,
                "person_ids": [None if pd.isna(v) else int(v) for v in group["person_id"]],
            })
    return violations


def build_windows(violations: list, margin: float, body: float = DEFAULT_BODY_SEC) -> list:
    """
    違反フレームを前後 margin 秒に広げ、重なる区間をまとめて時間窓のリストにする。

    違反は普通 1 フレームではなく数秒続くので、素朴に 1 フレーム 1 クリップにすると
    同じ場面の動画が何十本も出る。重なりをマージしてから上限で切る。

    1 本の長さは margin + body + margin で頭打ちにする。違反が 30 秒続いても
    原因が分かるのは序盤だけで、残りは同じ絵が延々続くだけだから。打ち切っても
    違反フレームの一覧は label_violations.txt に全件残る。

    マージの判定には打ち切り前の終端（raw_end）を使う。打ち切った終端で判定すると、
    続いている違反が「次の区間」と見なされて同じ場面のクリップが量産される。
    """
    delta = pd.to_timedelta(margin, unit="s")
    max_length = pd.to_timedelta(2 * margin + body, unit="s")
    windows = []
    raw_end = None
    for v in violations:
        ts = pd.to_datetime(v["timestamp"], format=TIME_FORMAT)
        start, end = ts - delta, ts + delta
        if windows and start <= raw_end:
            raw_end = max(raw_end, end)
            windows[-1] = (windows[-1][0], min(raw_end, windows[-1][0] + max_length))
        else:
            windows.append((start, end))
            raw_end = end
    return windows


def write_report(check_dir: str, data_name: str, violations: list, n_frames: int,
                 windows: list, margin: float, body: float, kept_windows: int) -> str:
    ratio = 100.0 * len(violations) / n_frames if n_frames else 0.0
    rule = ", ".join(f"{LABEL_NAMES[v]}<={n}" for v, n in sorted(MAX_PER_LABEL.items()))

    lines = [
        f"data={data_name}",
        f"rule={rule}",
        f"violation_frames={len(violations)} / {n_frames} ({ratio:.2f}%)",
        f"violation_windows={len(windows)} "
        f"(margin={margin}s, body={body}s, clips={kept_windows})",
        "",
        f"{'image':<30} {'labels(actual)':<28} person_ids",
    ]
    for v in violations:
        lines.append(
            f"{v['image']:<30} {_labels_text(v['labels']):<28} {v['person_ids']}")
    # if len(violations) > REPORT_ROWS:
        # lines.append(f"... 他 {len(violations) - REPORT_ROWS} 件")

    path = os.path.join(check_dir, "label_violations.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def check(data_name: str, df: pd.DataFrame, area_size: float = 2.0,
          margin: float = DEFAULT_MARGIN_SEC, body: float = DEFAULT_BODY_SEC,
          max_clips: int = DEFAULT_MAX_CLIPS,
          fps: int = 30, make_clips: bool = True) -> int:
    """
    ラベル個数のチェックを行い、違反フレーム数を返す（0 なら正常）。

    df は person_id / label と bbox 座標を持つ、ラベル付与直後の DataFrame。
    CSV に書き出す前の状態をそのまま渡せるようにしてある。
    """
    check_dir = os.path.join(processed_dir(data_name), "check")
    removed = clear_violation_files(check_dir)
    if removed:
        print(f"[verify] 前回の violation ファイル {removed} 件を削除しました")

    rule = ", ".join(f"{LABEL_NAMES[v]}<={n}" for v, n in sorted(MAX_PER_LABEL.items()))
    n_frames = df["timestamp"].nunique()
    violations = find_violations(df)
    if not violations:
        print(f"[verify] 同時刻のラベル個数チェック（{rule}）: "
              f"{n_frames} フレームすべて正常")
        return 0

    ratio = 100.0 * len(violations) / n_frames if n_frames else 0.0
    print(f"[verify] [error] 同時刻に存在してよい数（{rule}）を超えたフレームが "
          f"{len(violations)} / {n_frames} 件あります（{ratio:.2f}%）")
    for v in violations[:3]:
        print(f"[verify]         {v['image']}  labels={_labels_text(v['labels'])} "
              f"（超過: {','.join(v['exceeded'])}）  person_ids={v['person_ids']}")
    print(f"[verify]         YAML の persons.ids / start_phase の割り当てを見直してください")

    os.makedirs(check_dir, exist_ok=True)

    windows = build_windows(violations, margin, body)
    kept = windows[:max_clips]
    report = write_report(check_dir, data_name, violations, n_frames,
                          windows, margin, body, len(kept))
    print(f"[verify]         → {report} に一覧を書き出しました")

    if make_clips:
        if len(windows) > max_clips:
            print(f"[verify]         違反区間 {len(windows)} 個のうち先頭 {max_clips} 個の"
                  f"クリップを生成します（--max_clips で変更可）")
        # 描画ライブラリは重いのでここまで来て初めて読む
        import make_check_videos as mcv
        mark_ts = {v["timestamp"] for v in violations}
        try:
            _, results, _ = mcv.render_windows(
                data_name, area_size, kept, out_prefix="violation", fps=fps,
                bbox_override=df, bbox_text="id", mark_ts=mark_ts,
                mark_note="LABEL COUNT VIOLATION", log_prefix="[verify]")
            print(f"[verify]         → 検証クリップ {len(results)} 本を生成しました "
                  f"（bbox の上は person_id、違反フレームは赤枠）")
        except Exception as e:
            # クリップは付加物。ここで落ちても違反そのものは報告済みにする。
            print(f"[verify] [warn] 検証クリップの生成に失敗しました: {e}")

    return len(violations)


def run(data_name: str, area_size: float = 2.0,
        margin: float = DEFAULT_MARGIN_SEC, body: float = DEFAULT_BODY_SEC,
        max_clips: int = DEFAULT_MAX_CLIPS, fps: int = 30) -> int:
    """単体実行用。ラベル付け済みの bbox_label.csv を読んで検査する。"""
    bbox_path = os.path.join(processed_dir(data_name), "bbox_label.csv")
    if not os.path.exists(bbox_path):
        print(f"[verify] [error] bbox_label.csv が見つかりません: {bbox_path}")
        sys.exit(1)

    df = pd.read_csv(bbox_path)
    if "label" not in df.columns:
        print(f"[verify] [error] label 列がありません: {bbox_path}")
        sys.exit(1)
    # preprocess_real.py は label=0 で書き出すので、0 しか無いファイルはラベル付け前。
    # 1人・2人データだと walk が並ぶだけで素通りしてしまい、検査した気になりやすい。
    if (df["label"].fillna(0) != 0).sum() == 0:
        print(f"[verify] [warn] label が全て 0 です。ラベル付け前の bbox_label.csv を"
              f"見ている可能性があります（relabel.py が中断していないか確認してください）")

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format=TIME_FORMAT)
    df = df.sort_values("timestamp_dt")
    return check(data_name, df, area_size=area_size, margin=margin, body=body,
                 max_clips=max_clips, fps=fps)


def main():
    parser = argparse.ArgumentParser(
        description="同時刻のラベル個数チェック（複数人データの ids 割り当て検証）",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", required=True, help="データフォルダ名 例: threeperson_gym_01")
    parser.add_argument("--area_size", type=float, default=2.0,
                        help="検証クリップの 3D パネル軸範囲（前処理の --area_size と同じ値）")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN_SEC,
                        help=f"違反フレームの前後に取る秒数（既定: {DEFAULT_MARGIN_SEC}）")
    parser.add_argument("--body", type=float, default=DEFAULT_BODY_SEC,
                        help=f"違反区間から切り出す長さ・秒（既定: {DEFAULT_BODY_SEC}）。"
                             f"1 本の長さは margin + body + margin で頭打ちになる")
    parser.add_argument("--max_clips", type=int, default=DEFAULT_MAX_CLIPS,
                        help=f"生成するクリップ数の上限（既定: {DEFAULT_MAX_CLIPS}）")
    parser.add_argument("--fps", type=int, default=30, help="出力 fps（既定: 30）")
    args = parser.parse_args()

    n_bad = run(args.data, area_size=args.area_size, margin=args.margin,
                body=args.body, max_clips=args.max_clips, fps=args.fps)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
