"""
行動サイクル（1フェーズ phase_sec 秒 × N フェーズ）の共通定義。phase_sec の既定は 5 秒。

撮影プロトコル:
  gym     : walk, walk, stop, crouch, banzai            = 5 フェーズ / 25 秒
  meeting : walk, walk, stop, crouch, banzai, お辞儀     = 6 フェーズ / 30 秒

お辞儀は本来の実験プロトコルに含めるつもりがなかった動作なので、5 番目のクラスとして
学習させるのではなくフレームごと除外する（YAML の drop_phases で指定）。ラベルは
どちらのサイクルでも 4 クラス（walk / stop / crouch / hand）のまま。

位相の時刻原点は常に --start_img（しゃがみ始めの画像）の時刻。生き残った最初の行
（df.timestamp_dt.min()）を原点にすると、関節フィルタや人数フィルタ、お辞儀フレームの
除外で先頭フレームが落ちたときに原点がずれ、全ラベルがずれる。

複数人データの track_id -> 人物の割り当て（--persons の ids）は時間依存にできる。
ByteTrack の track_id は録画の途中で別人に乗り移ることがあり、「id 1 は前半 A さん・
後半 B さん」という状況が起きるため、id 単位の割り当てでは表現しきれない。ids の要素を
{"id": 1, "from": ..., "to": ...} と書くとその区間だけの割り当てになる（int で書けば
従来どおり全区間）。区間は from を含み to を含まない半開区間で、同じ時刻を前の区間の
to と次の区間の from に書けば穴も重なりも生じない。

収録中に行動サイクルを 1 回飛ばしてしまうと、その時刻以降の位相が丸ごとずれる。原点と
start_phase は録画を通して一定なのでこのずれは表現できない。shifts に
[{"from": <時刻>, "delta": 1}] と書くと「その時刻以降は位相を delta だけ進める」という
補正になる。persons の人物ごとに書けばその人物だけ、--shifts に書けば全行に効く
（1人モードや、全員が同時に飛ばした場合はこちら）。位相は最後に cycle_phases で丸める
ので、delta が負でも cycle_phases を超えても構わない。

このモジュールが持つのは「preprocess_real.py（お辞儀フレームの除外）と relabel.py
（ラベル付与）が一致していないと壊れるもの」だけ。どちらか一方でしか使わない処理は
置かない。どのフェーズを落とすかは YAML で与えるデータなので、PHASE_BOW のような
定数は意図的に定義していない。
"""
import os
import sys
import json

import pandas as pd

TIME_FORMAT = "%Y-%m-%d %H:%M:%S %f"

PHASE_DURATION_SEC = 5.0                          # 1 フェーズの秒数の既定値（--phase_sec）
DEFAULT_CYCLE_PHASES = 5                          # gym など従来データ
PHASE_TO_LABEL = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}   # 0,1 walk / 2 stop / 3 crouch / 4 hand
ONEPERSON_START_PHASE = 3                         # 1人モード: start_img = しゃがみ始め


def filename_to_timestamp(filename: str) -> str:
    """
    カラー画像のファイル名を timestamp 文字列に変換する。
    20260520_064853_041393.jpg -> "2026-05-20 06:48:53 041393"
    """
    name = os.path.splitext(os.path.basename(filename))[0]  # 拡張子除去
    parts = name.split("_")
    if len(parts) != 3:
        raise ValueError(f"ファイル名の形式が不正です: {filename}")
    date, time, microsec = parts
    return f"{date[:4]}-{date[4:6]}-{date[6:8]} {time[:2]}:{time[2:4]}:{time[4:6]} {microsec}"


def timestamp_to_filename(timestamp: str) -> str:
    """
    timestamp 文字列をカラー画像のファイル名に戻す（filename_to_timestamp の逆）。
    "2026-05-20 06:48:53 041393" -> 20260520_064853_041393.jpg
    """
    date_part, time_part, micro = str(timestamp).strip().split(" ")
    return (f"{date_part.replace('-', '')}_"
            f"{time_part.replace(':', '')}_{micro}.jpg")


def anchor_timestamp(start_img: str) -> pd.Timestamp:
    """位相の時刻原点（--start_img の時刻）を Timestamp で返す。"""
    return pd.to_datetime(filename_to_timestamp(start_img), format=TIME_FORMAT)


def parse_time_point(value, log_prefix: str = "[phase]") -> pd.Timestamp:
    """
    ids の区間指定（from / to）の端点を Timestamp に変換する。

    受け付けるのは 2 形式:
      - timestamp 列と同じ "2026-01-29 08:39:45 703176"
      - --start_img と同じカラー画像のファイル名／パス（20260129_083945_703176.jpg）
    違反クリップの RGB パネルに出るのは前者、raw に並んでいるのは後者なので、
    どちらを書き写しても通るようにしてある。
    """
    if isinstance(value, pd.Timestamp):
        return value
    text = str(value).strip()
    try:
        return pd.to_datetime(text, format=TIME_FORMAT)
    except ValueError:
        pass
    try:
        return anchor_timestamp(text)
    except ValueError:
        print(f"{log_prefix} [error] 区間の時刻として解釈できません: {value}")
        print(f"{log_prefix}         \"2026-01-29 08:39:45 703176\" 形式か "
              f"20260129_083945_703176.jpg 形式で指定してください")
        sys.exit(1)


def parse_persons_arg(value, log_prefix: str = "[phase]"):
    """--persons の JSON 文字列をパースする。未指定なら None（1人モード）。"""
    if not value:
        return None
    if isinstance(value, list):
        persons = value
    else:
        try:
            persons = json.loads(value)
        except json.JSONDecodeError as e:
            print(f"{log_prefix} [error] --persons の JSON 解析に失敗しました: {e}")
            sys.exit(1)
    if not isinstance(persons, list) or not persons:
        print(f"{log_prefix} [error] --persons は非空の配列にしてください")
        sys.exit(1)
    return persons


def parse_drop_phases_arg(value, log_prefix: str = "[phase]") -> list:
    """
    --drop_phases をリストに正規化する。

    YAML に `drop_phases:` と値なしで書かれた場合や未指定の場合は空リスト。
    スカラー（drop_phases: 5 のような書き間違い）はリストに包んで受け入れる。
    """
    if value is None or value == "" or value == "None":
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        phases = value
    else:
        try:
            phases = json.loads(value)
        except json.JSONDecodeError as e:
            print(f"{log_prefix} [error] --drop_phases の JSON 解析に失敗しました: {e}")
            sys.exit(1)
    if isinstance(phases, int):
        phases = [phases]
    if not isinstance(phases, list):
        print(f"{log_prefix} [error] --drop_phases は整数の配列にしてください: {value}")
        sys.exit(1)
    for p in phases:
        if not isinstance(p, int) or isinstance(p, bool):
            print(f"{log_prefix} [error] --drop_phases の要素は整数にしてください: {p}")
            sys.exit(1)
    return phases


def parse_shifts_arg(value, log_prefix: str = "[phase]", where: str = "--shifts") -> list:
    """
    位相補正（shifts）を [(Timestamp, delta), ...] の時刻昇順リストに正規化する。

    書き方は [{"from": <時刻>, "delta": 1}, ...]。from はその補正が効き始める時刻
    （その時刻を含む）で、ids の from / to と同じくカラー画像のファイル名でも
    timestamp 文字列でも書ける。delta を省略すると 1 -- サイクルを 1 回飛ばすと
    以降の位相は 1 つ先に進むので、既定値が「1 回飛ばした」に相当する。
    要素を複数書けば効果は累積する（2 回飛ばした / 途中で元に戻った、など）。
    """
    if value is None or value == "" or value == "None":
        return []
    if isinstance(value, (list, dict)):
        entries = value
    else:
        try:
            entries = json.loads(value)
        except json.JSONDecodeError as e:
            print(f"{log_prefix} [error] {where} の JSON 解析に失敗しました: {e}")
            sys.exit(1)
    if isinstance(entries, dict):
        # 1 件だけなら配列に包まずに書けるようにする。
        entries = [entries]
    if not isinstance(entries, list):
        print(f"{log_prefix} [error] {where} は配列にしてください: {value}")
        sys.exit(1)

    shifts = []
    for entry in entries:
        if not isinstance(entry, dict) or "from" not in entry:
            print(f"{log_prefix} [error] {where} の要素は "
                  f"{{\"from\": ..., \"delta\": 1}} にしてください: {entry}")
            sys.exit(1)
        unknown = set(entry) - {"from", "delta"}
        if unknown:
            print(f"{log_prefix} [error] {where} に未知のキーがあります: {sorted(unknown)}")
            sys.exit(1)
        delta = entry.get("delta", 1)
        if not isinstance(delta, int) or isinstance(delta, bool):
            print(f"{log_prefix} [error] {where} の delta は整数にしてください: {delta}")
            sys.exit(1)
        if delta == 0:
            print(f"{log_prefix} [error] {where} の delta が 0 です（補正になりません）: {entry}")
            sys.exit(1)
        shifts.append((parse_time_point(entry["from"], log_prefix), delta))

    shifts.sort(key=lambda s: s[0])
    return shifts


def _shifts_text(shifts) -> str:
    """位相補正をログ用の文字列にする。"""
    return " / ".join(f"{ts} から {delta:+d}" for ts, delta in shifts)


def _check_shifts(shifts, cycle_phases: int, log_prefix: str, where: str):
    """delta が 1 サイクルぶん（＝位相が変わらない）になっていないかを警告する。"""
    for ts, delta in shifts:
        if delta % cycle_phases == 0:
            print(f"{log_prefix} [warn] {where} の delta {delta} は cycle_phases "
                  f"{cycle_phases} の倍数なので位相は変わりません（{ts}）")


def _row_timestamps(df: pd.DataFrame) -> pd.Series:
    """行の時刻を Series で返す（前段で作った timestamp_dt があればそれを使う）。"""
    if "timestamp_dt" in df.columns:
        return df["timestamp_dt"]
    return pd.to_datetime(df["timestamp"], format=TIME_FORMAT)


def validate_cycle(cycle_phases: int, drop_phases: list, log_prefix: str = "[phase]",
                   phase_sec: float = PHASE_DURATION_SEC):
    """cycle_phases / drop_phases / phase_sec の整合を検証する。"""
    if not phase_sec > 0 or _phase_delta(phase_sec) <= pd.Timedelta(0):
        print(f"{log_prefix} [error] --phase_sec は正の秒数（μs 以上）にしてください: {phase_sec}")
        sys.exit(1)

    if not isinstance(cycle_phases, int) or cycle_phases < 1:
        print(f"{log_prefix} [error] --cycle_phases は 1 以上の整数にしてください: {cycle_phases}")
        sys.exit(1)

    for p in drop_phases:
        if not (0 <= p < cycle_phases):
            print(f"{log_prefix} [error] --drop_phases の値は 0〜{cycle_phases - 1} にしてください: {p}")
            sys.exit(1)
        if p in PHASE_TO_LABEL:
            # 0〜4 はラベルを持つ通常の行動。落とすのはほぼ確実に指定ミス。
            print(f"{log_prefix} [warn] phase {p} はラベル {PHASE_TO_LABEL[p]} を持つ行動です。"
                  f"本当に除外しますか？")

    # 残るフェーズがすべてラベルに変換できることを確認する。
    surviving = [p for p in range(cycle_phases) if p not in drop_phases]
    unmapped = [p for p in surviving if p not in PHASE_TO_LABEL]
    if unmapped:
        print(f"{log_prefix} [error] ラベルに対応付けられないフェーズが残ります: {unmapped}")
        print(f"{log_prefix}         drop_phases で除外するか PHASE_TO_LABEL を拡張してください")
        sys.exit(1)


def _parse_id_entry(entry, person_index: int, log_prefix: str):
    """
    ids の 1 要素を (track_id, from, to) に正規化する。

    - int          : その track_id は全区間その人物（従来の書き方）
    - {"id": int, "from": ..., "to": ...} : 指定した区間だけその人物
      from / to は省略可（省略 = 録画の端まで、または直前の from と組み合わせ）。
      from は含み、to は含まない半開区間なので、同じ時刻を前の区間の to と
      次の区間の from に書けば穴も重なりも生じない。
      同じ id を複数回書いたときの組み合わせ方は build_track_to_phase を参照。
    """
    if isinstance(entry, bool):
        print(f"{log_prefix} [error] persons[{person_index}] の ids に真偽値があります: {entry}")
        sys.exit(1)
    if isinstance(entry, int):
        return entry, None, None
    if not isinstance(entry, dict) or "id" not in entry:
        print(f"{log_prefix} [error] persons[{person_index}] の ids の要素は "
              f"track_id か {{\"id\": ..., \"from\": ..., \"to\": ...}} にしてください: {entry}")
        sys.exit(1)

    unknown = set(entry) - {"id", "from", "to"}
    if unknown:
        print(f"{log_prefix} [error] persons[{person_index}] の ids に未知のキーがあります: "
              f"{sorted(unknown)}")
        sys.exit(1)

    track_id = entry["id"]
    if not isinstance(track_id, int) or isinstance(track_id, bool):
        print(f"{log_prefix} [error] persons[{person_index}] の ids の id は整数にしてください: "
              f"{track_id}")
        sys.exit(1)

    lo = parse_time_point(entry["from"], log_prefix) if entry.get("from") is not None else None
    hi = parse_time_point(entry["to"], log_prefix) if entry.get("to") is not None else None
    return track_id, lo, hi


def _spans_overlap(a, b) -> bool:
    """半開区間 [from, to) 同士が重なるか。None は録画の端（±無限）とみなす。"""
    lo = max(x for x in (a[0], b[0], pd.Timestamp.min) if x is not None)
    hi = min(x for x in (a[1], b[1], pd.Timestamp.max) if x is not None)
    return lo < hi


def build_track_to_phase(persons: list, cycle_phases: int, log_prefix: str = "[phase]") -> dict:
    """
    persons（[{"ids": [...], "start_phase": int, "shifts": [...]}, ...]）から
    track_id -> [(from, to, start_phase, shifts), ...] の辞書を構築する。

    track_id は途中で別人に乗り移ることがある（ByteTrack の ID スイッチ）。
    そのため割り当ては track_id 単位ではなく (track_id, 時刻) 単位で持つ。
    区間を指定しない従来の書き方は (None, None) の 1 区間として扱うので、
    既存の YAML はそのまま動く。

    同じ人物の ids に同じ id を複数書いたときは、書いた順に「区間を開いて閉じる」と
    読む。from だけの要素が区間を開き、to だけの要素が直前に開いた区間を閉じる:

      ids: [{id: 29, from: A}, {id: 29, to: B}]   -> [A, B) の 1 区間
      ids: [{id: 29, to: A}, {id: 29, from: B}]   -> [始点, A) と [B, 終点) の 2 区間

    上は「A から B まで」、下は「A まで」と「B から」で、どちらも自然に読める形に
    なる。開いたまま閉じずに終わった区間は録画の終端まで伸ばす。

    - 同一 track_id の区間が重なっていたらエラー（人物をまたぐ場合も含む）。
    - start_phase は 0〜cycle_phases-1 の範囲を検証。
    - shifts（その人物が収録中にサイクルを飛ばした時刻と位相のずれ）は区間に添えて
      持ち回る。区間を分けて start_phase を書き直す代わりに使うもので、ids の書き方は
      いっさい変えずに済む。適用は _resolve_with_spans。
    """
    track_to_spans = {}

    def add_span(track_id, lo, hi, start_phase, shifts, person_index):
        if lo is not None and hi is not None and lo >= hi:
            print(f"{log_prefix} [error] persons[{person_index}] の id {track_id} の区間が "
                  f"from >= to になっています: {lo} 〜 {hi}")
            sys.exit(1)
        spans = track_to_spans.setdefault(track_id, [])
        for other in spans:
            if _spans_overlap((lo, hi), other):
                print(f"{log_prefix} [error] track_id {track_id} の割り当て区間が重複しています: "
                      f"{_span_text(lo, hi)} と {_span_text(*other[:2])}")
                print(f"{log_prefix}         同じ track_id を複数の人物に割り当てるときは "
                      f"from / to で時間を分けてください")
                sys.exit(1)
        spans.append((lo, hi, start_phase, shifts))

    for i, person in enumerate(persons):
        if "ids" not in person or "start_phase" not in person:
            print(f"{log_prefix} [error] persons[{i}] に ids / start_phase がありません: {person}")
            sys.exit(1)
        unknown = set(person) - {"ids", "start_phase", "shifts"}
        if unknown:
            print(f"{log_prefix} [error] persons[{i}] に未知のキーがあります: {sorted(unknown)}")
            sys.exit(1)
        start_phase = person["start_phase"]
        if not (isinstance(start_phase, int) and 0 <= start_phase < cycle_phases):
            print(f"{log_prefix} [error] persons[{i}] の start_phase は "
                  f"0〜{cycle_phases - 1} の整数にしてください: {start_phase}")
            sys.exit(1)

        shifts = parse_shifts_arg(person.get("shifts"), log_prefix, f"persons[{i}] の shifts")
        if shifts:
            _check_shifts(shifts, cycle_phases, log_prefix, f"persons[{i}] の shifts")
            print(f"{log_prefix} persons[{i}] の位相補正: {_shifts_text(shifts)}")

        open_from = {}   # track_id -> 閉じられるのを待っている下端
        for entry in person["ids"]:
            track_id, lo, hi = _parse_id_entry(entry, i, log_prefix)

            if lo is not None and track_id in open_from:
                print(f"{log_prefix} [error] persons[{i}] の id {track_id} は "
                      f"from が連続しています（{open_from[track_id]} が to で閉じられていません）")
                sys.exit(1)

            if lo is not None and hi is None:
                # from だけ: to が来るまで開いておく（来なければ録画の終端まで）
                open_from[track_id] = lo
                continue
            if hi is not None and lo is None:
                # to だけ: 直前に開いた区間があればその下端を使う
                lo = open_from.pop(track_id, None)

            add_span(track_id, lo, hi, start_phase, shifts, i)

        for track_id, lo in open_from.items():
            add_span(track_id, lo, None, start_phase, shifts, i)

    return track_to_spans


def _span_text(lo, hi) -> str:
    """区間をログ用の文字列にする。"""
    return f"[{'始点' if lo is None else lo} 〜 {'終点' if hi is None else hi})"


def _resolve_with_spans(df: pd.DataFrame, track_to_spans: dict,
                        log_prefix: str) -> pd.Series:
    """(person_id, 時刻) から start_phase を引く。区間は track ごとに数個なので行ループにならない。"""
    ts = _row_timestamps(df)

    out = pd.Series(float("nan"), index=df.index, dtype="float64")
    for track_id, spans in track_to_spans.items():
        is_id = df["person_id"] == track_id
        if not is_id.any():
            continue
        for lo, hi, start_phase, shifts in spans:
            mask = is_id
            if lo is not None:
                mask = mask & (ts >= lo)
            if hi is not None:
                mask = mask & (ts < hi)
            if not mask.any() and (lo is not None or hi is not None):
                # 区間を書いたのに 1 行も当たらない = 時刻の写し間違いがほぼ確実。
                # 区間なしの id が当たらないのは（フィルタで落ちただけの）普通のことなので黙る。
                print(f"{log_prefix} [warn] track_id {track_id} の区間 "
                      f"{_span_text(lo, hi)} に該当する行がありません"
                      f"（時刻の指定を確認してください）")
            out[mask] = start_phase
            # その人物の位相補正。区間の途中で効き始めることもあるので mask と AND する。
            for shift_at, delta in shifts:
                out[mask & (ts >= shift_at)] += delta
    return out


def resolve_start_phase(df: pd.DataFrame, persons, cycle_phases: int,
                        log_prefix: str = "[phase]", strict: bool = True,
                        shifts=()) -> pd.Series:
    """
    行ごとの start_phase を解決して Series で返す。

    shifts（--shifts）は全行に効く位相補正で、1人モードのずれや、全員が同時にサイクルを
    飛ばした場合に使う。人物ごとのずれは persons の各要素に shifts を書く。両方書いた
    場合は足し合わさる。補正後の値は 0〜cycle_phases-1 をはみ出すが、compute_phase が
    最後に剰余を取るので問題ない。
    """
    start_phase = _resolve_base_start_phase(df, persons, cycle_phases, log_prefix, strict)

    if shifts:
        _check_shifts(shifts, cycle_phases, log_prefix, "--shifts")
        print(f"{log_prefix} 全行への位相補正: {_shifts_text(shifts)}")
        ts = _row_timestamps(df)
        for shift_at, delta in shifts:
            # 未解決（NaN）の行はそのまま NaN。strict=False の呼び出し側が落とす。
            start_phase[ts >= shift_at] += delta

    return start_phase


def _resolve_base_start_phase(df: pd.DataFrame, persons, cycle_phases: int,
                              log_prefix: str = "[phase]", strict: bool = True) -> pd.Series:
    """
    位相補正（shifts）を当てる前の start_phase を返す。

    strict=True  : 未マップ / 欠損 person_id はエラー終了（relabel 用）。
    strict=False : [warn] を出して NaN のまま残す（preprocess_real 用）。関節フィルタの
                   前は「人が検出されなかった NaN 行」や一瞬だけ現れた track_id が大量に
                   残っており、ここで止めると YAML の ids を埋めるワークフローが壊れる。
                   YAML が不完全なままでも relabel 側で必ず落ちるので安全性は保たれる。

    ids に区間（from / to）が指定されていれば (person_id, 時刻) で引く。区間が
    ひとつも無ければ従来どおり map 一発で済ませる（全データの既定はこちら）。

    person_id は NaN 行の存在により float64 になるが、Series.map({1: ...}) は
    hash(1.0) == hash(1) なので int キーでも引ける。型変換は不要（触らないこと）。
    """
    if persons:
        track_to_spans = build_track_to_phase(persons, cycle_phases, log_prefix)
        timed = [tid for tid, spans in track_to_spans.items()
                 if any(lo is not None or hi is not None for lo, hi, _, _ in spans)]
        shifted = any(sh for spans in track_to_spans.values() for _, _, _, sh in spans)
        note = f" / 区間指定 {len(timed)} 件" if timed else ""
        print(f"{log_prefix} 複数人モード: {len(persons)} 人 / "
              f"track_id {len(track_to_spans)} 件{note}")

        if timed or shifted:
            start_phase = _resolve_with_spans(df, track_to_spans, log_prefix)
        else:
            # 区間指定も位相補正も無いなら従来の高速パス
            # （track_id -> start_phase の単純な map）
            flat = {tid: spans[0][2] for tid, spans in track_to_spans.items()}
            start_phase = df["person_id"].map(flat)

        present_ids = df["person_id"].dropna().unique()
        unmapped = [pid for pid in present_ids if pid not in track_to_spans]
        has_nan = bool(df["person_id"].isna().any())
        # ids には載っているが、どの区間にも入らない行。区間指定を入れたときだけ起きる
        # （from / to の穴）。未マップ id とは直し方が違うので分けて報告する。
        uncovered = df["person_id"].notna() & ~df["person_id"].isin(unmapped) & start_phase.isna()

        if strict:
            if unmapped:
                print(f"{log_prefix} [error] persons の ids に含まれない person_id があります: "
                      f"{sorted(unmapped)}")
                print(f"{log_prefix}         YAML の persons.ids を補完してください")
                sys.exit(1)
            if has_nan:
                print(f"{log_prefix} [error] person_id が欠損している行があります")
                sys.exit(1)
            if uncovered.any():
                sample = df.loc[uncovered, ["timestamp", "person_id"]].head(3).values.tolist()
                print(f"{log_prefix} [error] ids の区間に含まれない行が "
                      f"{int(uncovered.sum())} 件あります: {sample}")
                print(f"{log_prefix}         from / to の指定に隙間があります。"
                      f"区間を埋めるか、端の区間から from / to を外してください")
                sys.exit(1)
        else:
            if unmapped:
                print(f"{log_prefix} [warn] persons の ids に含まれない person_id: "
                      f"{sorted(unmapped)}（該当行はフェーズ判定から除外します）")
            if has_nan:
                n_nan = int(df["person_id"].isna().sum())
                print(f"{log_prefix} [warn] person_id が欠損している行が {n_nan} 件あります"
                      f"（フェーズ判定から除外します）")
            if uncovered.any():
                print(f"{log_prefix} [warn] ids の区間に含まれない行が "
                      f"{int(uncovered.sum())} 件あります（フェーズ判定から除外します）")

        return start_phase

    # 1人モード: person_id を自動取得し、開始フェーズはしゃがみ固定。
    person_ids = df["person_id"].dropna().unique()
    if len(person_ids) == 0:
        print(f"{log_prefix} [error] person_id が見つかりません")
        sys.exit(1)
    if len(person_ids) > 1:
        print(f"{log_prefix} [warn] 複数の person_id が存在します: {person_ids}")
        print(f"{log_prefix} [warn] 複数人データの場合は --persons を指定してください")
    print(f"{log_prefix} 1人モード: person_id = {person_ids[0]} "
          f"(start_phase = {ONEPERSON_START_PHASE})")

    # track_id が途中で変わっても 1人データは 1人なので、全行に同じ start_phase を入れる。
    # （最初の person_id だけを map すると他の track_id の行が NaN ラベルになる）
    return pd.Series(ONEPERSON_START_PHASE, index=df.index, dtype="int64")


def _phase_delta(phase_sec: float) -> pd.Timedelta:
    """1 フェーズの長さを μs 単位の Timedelta にする（0.1 のような小数でも誤差を持ち込まない）。"""
    return pd.Timedelta(microseconds=round(phase_sec * 1_000_000))


def compute_phase(ts: pd.Series, anchor: pd.Timestamp, start_phase: pd.Series,
                  cycle_phases: int, log_prefix: str = "[phase]",
                  phase_sec: float = PHASE_DURATION_SEC) -> pd.Series:
    """
    各行のフェーズ（0〜cycle_phases-1）を返す。

    算術の注意:
      - `//` してから `%` の順を変えないこと。(elapsed / phase_sec).astype(int) や int(x) は
        0 方向に切り捨てるため、原点より前の行で位相が壊れる。`//` は -inf 方向に
        floor し、pandas の `%` は除数の符号に従うので結果は常に [0, cycle) に入る。
      - 割り算は float 秒ではなく Timedelta 同士で行う。phase_sec が 0.1 のような小数だと
        float の `//` は境界ちょうどのフレームを 1 つ前のフェーズに落とすことがあるが、
        μs に丸めた Timedelta 同士の `//` は整数演算なので正確。イプシロンは不要。
    """
    if start_phase.isna().any():
        raise ValueError(
            f"{log_prefix} start_phase に NaN が含まれています。"
            f"NaN を許容する呼び出し側は compute_phase の前にマスクしてください")

    elapsed = (ts - anchor).dt.total_seconds()
    if len(elapsed) > 0 and elapsed.min() < 0:
        n_before = int((elapsed < 0).sum())
        print(f"{log_prefix} [warn] 時刻原点より前の行が {n_before} 件あります"
              f"（最小 {elapsed.min():.3f} 秒）。--start_img の指定を確認してください")

    base_cycle = ((ts - anchor) // _phase_delta(phase_sec)).astype("int64")
    return (base_cycle + start_phase.astype("int64")) % cycle_phases


def phase_to_label(phase: pd.Series, log_prefix: str = "[phase]") -> pd.Series:
    """フェーズを学習用の 4 クラスラベルに変換する。"""
    label = phase.map(PHASE_TO_LABEL)
    if label.isna().any():
        leaked = sorted(phase[label.isna()].unique())
        raise ValueError(
            f"{log_prefix} ラベルに変換できないフェーズが残っています: {leaked}。"
            f"drop_phases での除外が効いていない可能性があります")
    return label.astype("int64")
