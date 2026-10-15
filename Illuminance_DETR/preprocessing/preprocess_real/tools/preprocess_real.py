import pandas as pd
import csv
import argparse
import sys
from datetime import datetime, timedelta
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import raw_dir, processed_dir
from phase import (DEFAULT_CYCLE_PHASES, PHASE_DURATION_SEC, TIME_FORMAT, anchor_timestamp, compute_phase,
                   parse_drop_phases_arg, parse_persons_arg, parse_shifts_arg,
                   resolve_start_phase,
                   validate_cycle)
from sensor_layout import rename_to_ch
# 関節の定義は学習側と共有する（util/keypoints.py が唯一の定義元）。
# parents: [0]=tools [1]=preprocess_real [2]=preprocessing [3]=Illuminance_DETR [4]=repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Illuminance_DETR.util import keypoints

# 水平方向(X/Y)の bbox 固定サイズ（メートル）。高さ(Z)は関節 Z の robust min/max で動的。
# 実データを見て調整すること。--bbox_size で上書きできる。
BBOX_SIZE_M = 0.5

# 同一 timestamp 内で「同一人物の二重検出」とみなす関節間距離（メートル）。
# 座標はメートル化済みなので小さい値にする（別人同士を重複と誤判定しないため）。
DUPLICATE_THRESHOLD_M = 0.1

# pose_origin.csv（YOLO の生値）に掛ける時系列平滑化の既定値。
SMOOTH_MEDIAN_WINDOW = 5    # depth スパイク除去用の rolling median 窓（フレーム数）
SMOOTH_MEAN_WINDOW = 3      # 平滑化用の rolling mean 窓（フレーム数）
# 欠損の長さは「直前の有効値から直後の有効値までの時間差」で測る。実データは 33.34 ms/frame
# なので、N フレーム連続欠損 = (N+1) 区間。10 フレーム欠損 = 11 区間 = 0.367 秒で、
# 余裕を見て 0.38 秒を既定とする。
#
# この値は実測に基づいて決めた。旧既定の 0.15 秒はちょうど 3F(0.133s) と 4F(0.167s) の境目に
# あたり、pose.csv の欠損が「1〜3F はほぼ皆無、4F で急増」という不自然な分布になっていた。
# 一方、実際の欠損箇所と同じ動きの大きさに条件付けた補間誤差は 10F でも中央値 3.6 cm
# （90%点 5.5 cm）で、モデルの MPJPE 144〜249 mm に対して十分小さい。
# 20F まで伸ばすと 6.6 cm（MPJPE の 26〜46%）になるため、そこまでは広げない。
SMOOTH_MAX_GAP_SEC = 0.38

# 「これ以上離れた行同士は別区間」として平滑化をまたがせない閾値。補間の上限とは
# 役割が違うので分けている。ここを補間側と同じ 0.38 秒まで広げると、人物が丸ごと
# 未検出で行自体が無い箇所を「連続」とみなし、rolling median/mean が 0.38 秒離れた
# 行同士を平均して実際の動きをぼかしてしまう。
# 行が 33 ms 間隔で連続している通常の欠損（行はあるが座標が NaN）では区間分割は
# 起きないので、この値を据え置いても補間による回復量には影響しない。
SMOOTH_SEGMENT_GAP_SEC = 0.15

# 隣接関節間距離（骨長）による関節フィルタの既定値。
# 関節のピクセル位置が数 px ずれて背景の depth を拾うと、その関節だけが奥に飛ぶ。
# detect_yolo.py の判定は conf >= 0.2 と depth > 0 だけなので、この取り違えは素通りする。
# 人体の骨長はフレーム間でほぼ一定なので、隣接関節間の距離で名指しで検出できる。
#
# 判定は「長すぎ」側と「短すぎ」側で非対称にしてある。単眼 depth では、骨が視線方向を
# 向くと両端が同じピクセルに乗って距離が潰れるため、「短すぎ」は測定原理上の当たり前の
# 現象として大量に出る。しかもその向きは動作と相関するので、対称に判定すると特定の動作
# クラスだけが消える（oneperson_gym_01 実測: 中央値から -35% 以上短い骨は crouch フェーズの
# 97.9% に出る = しゃがみクラスが全滅する。walk でも肩幅・腰幅が約 30%）。
# 一方「長すぎ」側は、背景の depth を拾って関節が奥に飛んだ場合にしか起きず、
# 解剖学的にもあり得ない。よって既定では長すぎ側だけを落とす。
#
# 基準長からの相対許容幅（長すぎ側）。中央値の 1.8 倍を超えた骨を異常とみなす。
# 実測ではこの設定で落ちるのは脛 0.63 m 超のような明確な破綻だけで、全体の 2.0%。
JOINT_DIST_TOL = 0.8
# 相対幅と併用する絶対許容幅（メートル）。判定は ref + max(tol * ref, slack) で行う。
# 関節 1 点の深度の不確かさが数 cm あり、骨長はその 2 点分を含むので、これより小さい
# 許容幅を指定しても測定精度を超えた判定になる。実際 0.02 m にすると鼻-目（4.4 cm）や
# 目-耳（7.1 cm）が偽陽性の主因になり、違反が動作フェーズと強く相関してしまう。
JOINT_DIST_SLACK_M = 0.10
# 「短すぎ」側の相対許容幅。None で無効（既定）。上記のとおり動作クラスと交絡するので、
# 有効にする場合はフェーズ別の除去率を必ず確認すること。0.95 程度まで上げれば
# 「完全に潰れた骨」だけを拾えるが、それでも walk に偏る。
JOINT_DIST_SHORT_TOL = None
# 人物ごとの基準長を信用する最小有効フレーム数。これを下回る人物（一瞬だけ映り込んだ
# track など）はデータ全体の中央値にフォールバックする。それも取れない骨は判定しない。
JOINT_DIST_MIN_SAMPLES = 30
# 1 行あたりに NaN 化する関節数の上限。ここまで壊れた行はどのみち後段の関節フィルタ
# （両腰 + 他 10 関節すべて）で落ちるので、貪欲法をこれ以上回しても意味が無い。
JOINT_DIST_MAX_DROP = 4

def parse_hhmmss_to_seconds(time_str):
    """
    Converts a string in 'hhmmss.ff' or '-hhmmss.ff' format to seconds (float).
    Custom function used as an argparse type.
    """
    try:
        is_negative = time_str.startswith('-')
        if is_negative:
            time_str = time_str[1:]

        if '.' in time_str:
            parts = time_str.split('.')
            main_part = parts[0]
            frac_part = parts[1]
        else:
            main_part = time_str
            frac_part = "0"

        main_part = main_part.zfill(6)
        
        hours = int(main_part[0:2])
        minutes = int(main_part[2:4])
        seconds = int(main_part[4:6])
        
        frac_seconds = float(f"0.{frac_part}")

        total_seconds = hours * 3600 + minutes * 60 + seconds + frac_seconds

        if is_negative:
            total_seconds *= -1
        
        return total_seconds
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"Invalid time format: '{time_str}'. Please specify in 'hhmmss.ff' or '-hhmmss.ff' format."
        )

def remove_duplicate_pose_frames(df, threshold=0.1):
    """
    If there are rows with nearly identical coordinates within the same timestamp, delete that timestamp.
    threshold: If the absolute difference in coordinates (in metres, after process_pose_df) is less
    than this value, they are considered 'nearly identical'. Keep it small so genuinely different
    people in a multi-person frame are not mistaken for a duplicate of one person.
    """
    print(f"   └ [Additional Step] Starting duplicate pose detection and timestamp removal (Threshold: {threshold})...")
    
    coord_cols = [col for col in df.columns if col.endswith(('_X', '_Y', '_Z'))]
    if len(coord_cols) % 3 != 0:
        print("WARNING: The number of coordinate columns is not a multiple of 3. Accurate determination may not be possible.")
    
    grouped = df.groupby('timestamp')
    timestamps_to_remove = []

    for ts, group in grouped:
        if len(group) > 1:
            vals = group[coord_cols].values
            is_duplicate_frame = False
            
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    diff = np.abs(vals[i] - vals[j])
                    joint_diffs = diff.reshape(-1, 3)
                    matched_joints = np.all(joint_diffs < threshold, axis=1)
                    if np.sum(matched_joints) >= 2:
                        is_duplicate_frame = True
                        break
            
            if is_duplicate_frame:
                timestamps_to_remove.append(ts)

    if timestamps_to_remove:
        original_ts_count = df['timestamp'].nunique()
        df = df[~df['timestamp'].isin(timestamps_to_remove)].copy()
        print(f"Removed {len(timestamps_to_remove)} timestamps due to duplicates.")
        print(f"(Total frames: {original_ts_count} -> {df['timestamp'].nunique()})")
    else:
        print("No timestamps with duplicate poses were found.")
        
    return df

def _bone_reference_lengths(lengths, logical, person, min_samples):
    """
    人物ごと・論理骨ごとの基準長（中央値）を各行に展開した DataFrame を返す。

    左右対称な骨は SYMMETRIC_BONES で 1 つの論理骨にまとめてあるので、基準長は左右で共有する。
    有効サンプルが min_samples に満たない人物（一瞬だけ映り込んだ track など）は、その骨だけ
    データ全体の中央値にフォールバックする。全体でも 1 点も取れない骨は NaN のままで、
    以降の判定から自動的に外れる。
    """
    names = sorted(set(logical))
    cols_of = {name: [i for i, n in enumerate(logical) if n == name] for name in names}

    global_ref = {}
    for name in names:
        vals = lengths[cols_of[name]].to_numpy(dtype=float).ravel()
        vals = vals[~np.isnan(vals)]
        global_ref[name] = float(np.median(vals)) if vals.size else np.nan

    ref = pd.DataFrame(np.nan, index=lengths.index, columns=lengths.columns, dtype=float)
    per_person = {}
    # person_id が NaN の行（誰も検出されなかったフレーム）は groupby が自動的に除外する。
    for pid, group in lengths.groupby(person, sort=False):
        resolved = {}
        for name in names:
            vals = group[cols_of[name]].to_numpy(dtype=float).ravel()
            vals = vals[~np.isnan(vals)]
            resolved[name] = float(np.median(vals)) if vals.size >= min_samples else global_ref[name]
        per_person[pid] = resolved
        for name in names:
            ref.loc[group.index, cols_of[name]] = resolved[name]

    return ref, global_ref, per_person


def _bone_violations(lengths, ref, tol, slack_m, short_tol):
    """
    骨長が基準から外れている箇所を True で返す。

    長すぎ側と短すぎ側は非対称に扱う（理由は JOINT_DIST_TOL のコメント参照）。
    短すぎ側は short_tol が None なら判定しない。
    """
    too_long = lengths > ref + np.maximum(ref * tol, slack_m)
    if short_tol is None:
        return too_long.fillna(False)
    too_short = lengths < ref - np.maximum(ref * short_tol, slack_m)
    return (too_long | too_short).fillna(False)


def _drop_displaced_face(out, person, tol, slack_m, short_tol, min_samples):
    """
    head（取れている顔関節の平均）と neck（両肩の中点）の距離が基準から外れている行の
    顔 5 点をまとめて NaN 化する。変更は out にインプレースで行い、対象行のマスクを返す。

    顔 5 点は RAW_SKELETON では耳-肩の 2 本でしか体幹につながっていない。顔がまとめて
    背景の depth に飛んだ場合、耳を落とした時点で鼻・目を縛る骨が無くなり、ずれたまま
    生き残る。学習側が使うのは顔の点そのものではなく head なので、head-neck の距離で
    最終確認し、外れていれば head の材料をすべて捨てる（= 後段で行ごと落ちる）。
    どの顔関節が生き残っているかに依らず判定できるのが、個々の骨長チェックとの違い。
    """
    face_avail = np.stack([keypoints.joint_available(out, j).to_numpy()
                           for j in keypoints.FACE_JOINTS], axis=1)
    face_xyz = np.stack([out[keypoints.joint_cols(j)].to_numpy(dtype=float)
                         for j in keypoints.FACE_JOINTS], axis=1)
    face_xyz = np.where(face_avail[:, :, None], face_xyz, np.nan)
    n_face = face_avail.sum(axis=1)
    # nansum は全 NaN でも警告を出さない（nanmean と違って 0 を返す）ので割り算で平均する。
    head = np.where(n_face[:, None] > 0,
                    np.nansum(face_xyz, axis=1) / np.maximum(n_face, 1)[:, None], np.nan)

    shoulders_ok = (keypoints.joint_available(out, 'left_shoulder')
                    & keypoints.joint_available(out, 'right_shoulder')).to_numpy()
    neck = (out[keypoints.joint_cols('left_shoulder')].to_numpy(dtype=float)
            + out[keypoints.joint_cols('right_shoulder')].to_numpy(dtype=float)) / 2
    neck = np.where(shoulders_ok[:, None], neck, np.nan)

    lengths = pd.DataFrame({0: np.sqrt(((head - neck) ** 2).sum(axis=1))}, index=out.index)
    if lengths[0].notna().sum() == 0:
        return None

    ref, global_ref, _ = _bone_reference_lengths(lengths, ["head_neck"], person, min_samples)
    violated = _bone_violations(lengths, ref, tol, slack_m, short_tol)[0]

    n_rows = int(violated.sum())
    print(f"   head_neck       {global_ref['head_neck']:.3f}  violations: {n_rows}")
    if n_rows:
        idxs = out.index[violated]
        for joint in keypoints.FACE_JOINTS:
            out.loc[idxs, keypoints.joint_cols(joint)] = np.nan
        print(f"   → dropped all {len(keypoints.FACE_JOINTS)} face joints in {n_rows} rows "
              "(head would have been displaced)")
    return violated


def filter_joints_by_bone_length(df, tol=JOINT_DIST_TOL, slack_m=JOINT_DIST_SLACK_M,
                                 short_tol=JOINT_DIST_SHORT_TOL,
                                 min_samples=JOINT_DIST_MIN_SAMPLES,
                                 max_drop=JOINT_DIST_MAX_DROP):
    """
    隣接関節間の距離（骨長）が人物ごとの基準長から外れている関節を NaN 化する。

    深度の取り違え（関節のピクセル位置が背景に乗り、その関節だけ奥に飛ぶ）は
    detect_yolo.py の conf / depth 判定を素通りし、もっともらしい数値のまま残る。
    人体の骨長はフレーム間でほぼ一定なので、隣接関節間の距離を測れば名指しで特定できる。

    ただし判定は「長すぎ」側だけが既定で有効になっている。単眼 depth では視線方向を向いた
    骨の距離が原理的に潰れ、その向きは動作と相関するため、「短すぎ」側を有効にすると
    特定の動作クラスだけが大量に消える（詳細は JOINT_DIST_TOL のコメント）。

    判定は座標がメートルであることに依存するので、process_pose_df（X/Y を軸ごとに別倍率で
    リスケールする）より前に呼ぶこと。さらに smooth_pose_df の前に置くことで、
      - 外れ値が rolling mean で近傍フレームに拡散するのを防ぎ、
      - ここで空けた穴を _fill_short_gaps が補間で埋め直す
    という二重の利点がある。補間で戻せなかった関節だけが後段の関節フィルタで行ごと落ちる。

    犯人の特定は貪欲法で行う。違反している骨の端点を数え、最も多くの違反に関与している関節を
    1 つ NaN 化する。その関節に接する骨は評価不能になって違反集合から消えるので、違反が
    無くなるか max_drop に達するまで繰り返す。同数で並んだ場合は JOINT_DROP_PRIORITY が
    大きい方（末端・顔）を選び、体幹に近い関節を守る。
    """
    short_desc = f"{short_tol:.0%} short" if short_tol is not None else "short side off"
    print(f"   └ [Additional Step] Filtering joints by bone length "
          f"(tolerance: +{tol:.0%} long / {short_desc}, at least {slack_m} m, "
          f"max {max_drop} joints/row)...")

    bones = [(a, b) for a, b in keypoints.RAW_SKELETON
             if all(c in df.columns for c in keypoints.joint_cols(a) + keypoints.joint_cols(b))]
    if not bones:
        print("WARNING: No joint columns usable for the bone length check. Skipping.")
        return df

    skipped = len(keypoints.RAW_SKELETON) - len(bones)
    if skipped:
        print(f"WARNING: {skipped} of {len(keypoints.RAW_SKELETON)} bones skipped (columns missing).")

    out = df.copy()
    lengths = pd.DataFrame({i: keypoints.bone_length(out, a, b) for i, (a, b) in enumerate(bones)},
                           index=out.index)
    logical = [keypoints.SYMMETRIC_BONES.get(bone, f"{bone[0]}-{bone[1]}") for bone in bones]
    person = out['person_id'] if 'person_id' in out.columns else pd.Series(0, index=out.index)

    ref, global_ref, per_person = _bone_reference_lengths(lengths, logical, person, min_samples)

    violated = _bone_violations(lengths, ref, tol, slack_m, short_tol)

    # 違反が 1 本でもある行だけを回す（通常は全体の数 % なので全行ループより十分速い）。
    target_rows = violated.index[violated.any(axis=1)]
    rows_capped = 0
    to_nan = {}
    for row_idx in target_rows:
        bad = set(np.flatnonzero(violated.loc[row_idx].to_numpy()))
        dropped = []
        while bad and len(dropped) < max_drop:
            counts = {}
            for bone_idx in bad:
                for joint in bones[bone_idx]:
                    counts[joint] = counts.get(joint, 0) + 1
            culprit = max(counts,
                          key=lambda j: (counts[j], keypoints.JOINT_DROP_PRIORITY.get(j, 0)))
            dropped.append(culprit)
            bad = {bone_idx for bone_idx in bad if culprit not in bones[bone_idx]}
        if bad:
            rows_capped += 1
        for joint in dropped:
            to_nan.setdefault(joint, []).append(row_idx)

    for joint, idxs in to_nan.items():
        out.loc[idxs, keypoints.joint_cols(joint)] = np.nan

    # 基準長は tol を調整する唯一の手がかりなので必ず出す（前腕 ≈ 0.25 m, 肩幅 ≈ 0.35-0.40 m
    # あたりに乗っていれば自己校正は成功している）。
    print("Reference bone lengths (metres, median per person):")
    for name in sorted(set(logical)):
        per = [p[name] for p in per_person.values() if not np.isnan(p[name])]
        spread = f" [{min(per):.3f}-{max(per):.3f}]" if len(per) > 1 else ""
        n_viol = int(violated[[i for i, n in enumerate(logical) if n == name]].to_numpy().sum())
        print(f"   {name:<15} {global_ref[name]:.3f}{spread}  violations: {n_viol}")

    # 顔 5 点は耳-肩の 2 本でしか体幹につながっていないので、顔がまとめて背景の depth に
    # 飛ぶと、耳を落とした時点で鼻・目を縛るものが無くなり、ずれたまま生き残ってしまう。
    # 学習側が使うのは顔の点そのものではなく「取れている顔関節の平均 = head」なので、
    # ここで head-neck（SKELETON にもある辺）を測り、外れていれば顔 5 点を丸ごと落とす。
    # 単独でおかしい顔関節は上の貪欲法が先に除いているので、この判定は残った点で行われる。
    _drop_displaced_face(out, person, tol, slack_m, short_tol, min_samples)

    # 集計は入力と出力の差分から取る。貪欲法で落とした耳を顔パスが再度落とすことがあるので、
    # 各パスでの加算を足し合わせると二重に数えてしまう。
    checked = sorted({j for bone in bones for j in bone} | set(keypoints.FACE_JOINTS))
    nan_counts = {}
    newly_nan = pd.Series(False, index=out.index)
    for joint in checked:
        lost = keypoints.joint_available(df, joint) & ~keypoints.joint_available(out, joint)
        if lost.any():
            nan_counts[joint] = int(lost.sum())
            newly_nan |= lost
    rows_touched = int(newly_nan.sum())

    total_nan = sum(nan_counts.values())
    if total_nan:
        detail = ", ".join(f"{j}: {n}" for j, n in sorted(nan_counts.items(), key=lambda kv: -kv[1]))
        print(f"Bone length filter: invalidated {total_nan} joints in {rows_touched} rows "
              f"(of {len(out)}).")
        print(f"({detail})")
        if rows_capped:
            print(f"NOTE: {rows_capped} rows still violated after the {max_drop}-joint cap; "
                  "those rows are left to the joint filter below.")
    else:
        print("Bone length filter: no joint violated the constraint.")

    return out


def _fill_short_gaps(block, ts, max_gap_sec):
    """
    Fill only the SHORT holes of a coordinate block by time-linear interpolation.
    A hole is measured as (time of the first valid sample after it) - (time of the last valid sample
    before it); holes longer than max_gap_sec are left as NaN so the joint-count filter can drop
    those rows. Leading / trailing NaNs are never filled (that would be extrapolation).
    """
    na = block.isna()
    if not na.values.any():
        return block

    t = pd.DataFrame(np.repeat(ts.values[:, None], block.shape[1], axis=1),
                     index=block.index, columns=block.columns)
    t_valid = t.where(~na)
    hole_sec = (t_valid.bfill() - t_valid.ffill()).apply(lambda c: c.dt.total_seconds())

    interp = block.copy()
    interp.index = pd.DatetimeIndex(ts.values)
    interp = interp.interpolate(method='time', limit_area='inside')
    interp.index = block.index

    fillable = na & (hole_sec <= max_gap_sec)
    return block.where(~fillable, interp)


def smooth_pose_df(df, median_window=SMOOTH_MEDIAN_WINDOW, mean_window=SMOOTH_MEAN_WINDOW,
                   max_gap_sec=SMOOTH_MAX_GAP_SEC, segment_gap_sec=SMOOTH_SEGMENT_GAP_SEC):
    """
    Smooth the RAW joint coordinates of pose_origin.csv along time, per person.

    Order (per person, per continuous time segment):
      1. interpolate holes shorter than max_gap_sec (longer ones stay NaN -> dropped later)
      2. rolling median  : removes depth spikes
      3. rolling mean    : smooths the remaining jitter
    Both windows are centred, so no time lag is introduced against the (time-synced) light data.

    Two different gap thresholds are used on purpose:
      max_gap_sec     : how long a hole of NaN coordinates may be and still get interpolated.
      segment_gap_sec : how far apart two ROWS of the same person may be and still be smoothed
                        together. Rows further apart start a new segment, so a tracking dropout
                        (the person was not detected at all, hence no row exists) is never
                        averaged across. Keeping this small stops the rolling windows from
                        blending positions that are far apart in real time.

    The NaN mask taken after step 1 is re-applied at the end: `rolling` silently skips NaNs inside
    its window and would otherwise invent values for undetected joints — exactly the problem that
    made the old detect_yolo.py moving average hide missing detections.
    """
    print(f"[2/5] Smoothing Pose data (median window = {median_window}, mean window = {mean_window}, "
          f"max gap = {max_gap_sec} s, segment gap = {segment_gap_sec} s)...")
    coord_cols = [c for c in df.columns if c.endswith(('_X', '_Y', '_Z'))]
    if not coord_cols:
        print("WARNING: No coordinate columns found. Skipping smoothing.")
        return df

    out = df.copy()
    out[coord_cols] = out[coord_cols].astype(float)
    ts_all = pd.to_datetime(out['timestamp'], format='%Y-%m-%d %H:%M:%S %f')

    na_before = int(out[coord_cols].isna().sum().sum())

    # person_id が NaN の行は「誰も検出されなかったフレーム」で全座標が NaN なので対象外。
    person = out['person_id'] if 'person_id' in out.columns else pd.Series(0, index=out.index)
    keys = pd.DataFrame({'_ts': ts_all, '_person': person}).dropna(subset=['_person'])
    keys = keys.sort_values(['_person', '_ts'], kind='mergesort')
    dt = keys.groupby('_person', sort=False)['_ts'].diff().dt.total_seconds()
    is_jump = (dt > segment_gap_sec).fillna(False).astype(int)
    keys['_seg'] = is_jump.groupby(keys['_person'], sort=False).cumsum()

    filled_cells = 0
    n_segments = 0
    for _, g in keys.groupby(['_person', '_seg'], sort=False):
        n_segments += 1
        idx = g.index
        block = out.loc[idx, coord_cols]

        filled = _fill_short_gaps(block, g['_ts'], max_gap_sec)
        filled_cells += int(block.isna().values.sum() - filled.isna().values.sum())

        valid = filled.notna()
        smoothed = filled
        if median_window > 1:
            smoothed = smoothed.rolling(median_window, center=True, min_periods=1).median()
        if mean_window > 1:
            smoothed = smoothed.rolling(mean_window, center=True, min_periods=1).mean()
        out.loc[idx, coord_cols] = smoothed.where(valid)

    na_after = int(out[coord_cols].isna().sum().sum())
    print(f"Smoothing complete. Segments: {n_segments}, interpolated cells: {filled_cells}, "
          f"missing cells: {na_before} -> {na_after}")
    return out


def drop_phase_frames(df, persons, start_img, cycle_phases, drop_phases, shifts=(),
                      phase_sec=PHASE_DURATION_SEC):
    """
    Drop every frame in which ANY person is in one of `drop_phases`.

    Used for the meeting recordings, whose protocol has a 6th 5-second phase (a bow) that was
    not meant to be part of the experiment. The whole timestamp is removed, not just the bowing
    person's row: pose.csv only keeps frames with exactly `--people` rows, so removing a single
    row would make the frame fail the people filter anyway.

    Called BEFORE process_pose_df on purpose. process_pose_df -> _rescale_axis is the only step
    that derives a global statistic (the IQR-robust min/max of the ankle spread, fitted
    to area_size); frames we are about to discard must not influence that scale. Doing the drop
    afterwards in a separate step would leave the rescale fitted on the bow frames.

    `shifts` は収録中にサイクルを飛ばしたときの位相補正（--shifts）。ここで使う値は
    relabel.py に渡すものと必ず同じにする（食い違うと、除外したフェーズが relabel 側に
    残っていると言われて落ちる）。
    """
    if not drop_phases:
        print(f"   └ [Additional Step] Phase-based frame drop: skipped (no --drop_phases)")
        return df

    print(f"   └ [Additional Step] Dropping frames whose phase is in {drop_phases} "
          f"(cycle = {cycle_phases} phases x {phase_sec:g} s)...")

    start_phase = resolve_start_phase(df, persons, cycle_phases, "[preprocess]",
                                      strict=False, shifts=shifts)

    # start_phase が解決できなかった行（未マップ / 欠損 person_id）はフェーズ判定の
    # 根拠にしない。後続の関節フィルタでどのみち落ちる。
    valid = start_phase.notna()
    if not valid.any():
        print(f"[error] フェーズを判定できる行がありません。--persons の指定を確認してください")
        sys.exit("Failed to process Pose data.")

    ts = pd.to_datetime(df.loc[valid, 'timestamp'], format=TIME_FORMAT)
    phase = compute_phase(ts, anchor_timestamp(start_img), start_phase[valid],
                          cycle_phases, "[preprocess]", phase_sec)

    bad_ts = df.loc[phase[phase.isin(drop_phases)].index, 'timestamp'].unique()

    frames_before = df['timestamp'].nunique()
    rows_before = len(df)
    df = df[~df['timestamp'].isin(bad_ts)].copy()
    frames_after = df['timestamp'].nunique()

    ratio = (frames_before - frames_after) / frames_before * 100 if frames_before else 0.0
    print(f"Phase drop complete. Frames: {frames_before} -> {frames_after} "
          f"(dropped {frames_before - frames_after}, {ratio:.1f}%)")
    print(f"(Rows: {rows_before} -> {len(df)})")

    if len(df) == 0:
        print(f"[error] お辞儀フレームの除外で全行が消えました。--start_img / --persons / "
              f"--cycle_phases の指定を確認してください")
        sys.exit("Failed to process Pose data.")

    return df


def load_light_with_ch_header(input_csv_path):
    """
    Reads light_origin.csv (or environment_origin.csv) and returns a DataFrame whose sensor
    columns are named by sensor ID (ch name). The first timestamp column remains unchanged.

    座標はここでは作らない（[7.5] sensor_coords.py が sensor_coords.json に書く）。
    旧版 setup_data.py が "gx,gy" に書き換えた light_origin.csv も ch 名に戻して読む。
    """
    print(f"[1/3] Reading sensor data with ID header: {os.path.basename(input_csv_path)}")
    try:
        df = pd.read_csv(input_csv_path)
        df.columns, n_renamed = rename_to_ch(df.columns.tolist(), "Error:")
        if n_renamed:
            print(f"   └ Renamed {n_renamed} grid-style columns to sensor IDs.")
        print("Header check complete.")
        return df

    except Exception as e:
        print(f"Error: An error occurred while reading the sensor header: {e}")
        return None

def shift_light_timestamp(df, shift_seconds):
    """
    Shifts the timestamp column of the DataFrame by the specified number of seconds.
    """
    print(f"[2/3] Changing Light data timestamps... ({shift_seconds:+.1f} seconds)")
    try:
        original_format = '%Y-%m-%d %H:%M:%S %f'
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format=original_format)
        df['timestamp_dt'] = df['timestamp_dt'] + pd.to_timedelta(shift_seconds, unit='s')
        df['timestamp'] = df['timestamp_dt'].dt.strftime(original_format)
        df = df.drop(columns=['timestamp_dt'])
        
        print(f"Timestamp modification complete.")
        return df
        
    except Exception as e:
        print(f"Error: An error occurred during timestamp modification: {e}")
        return None

def _floor_reference(processed_df, axis, mode):
    """
    床面リスケールの基準にする系列を作る。axis は 'X' / 'Y'（Y<->Z スワップ後の床面）。

    mode='ankle'（既定）は左右足首の値をそのまま縦に積んだ系列。この系列の範囲が
    「人が床の上で占有した範囲」に対応するので、--area_size に合わせる対象として
    物理的な意味がはっきりする。
    mode='hip' は旧挙動（腰中点の軌跡）。過去データの再現用。
    """
    if mode == 'hip':
        return (processed_df[f'left_hip_{axis}'].astype(float)
                + processed_df[f'right_hip_{axis}'].astype(float)) / 2

    cols = [f'left_ankle_{axis}', f'right_ankle_{axis}']
    missing = [c for c in cols if c not in processed_df.columns]
    if missing:
        print(f"   └ [warn] {missing} が無いため、この軸だけ hip 基準にフォールバックします")
        return _floor_reference(processed_df, axis, 'hip')
    return pd.concat([processed_df[c].astype(float) for c in cols], ignore_index=True)


def _rescale_axis(processed_df, cols, ref_series, target_span):
    """
    Fit ref_series' robust (IQR) min/max to `target_span` and apply the SAME affine
    (centre-subtract then scale) to every column in `cols`, so the skeleton keeps its
    shape. Output is centred on the origin: [-target_span/2, +target_span/2].
    If the reference barely moves (span ~ 0) or target_span is non-positive, fall back to
    translation only (scale=1).
    Returns (df, center, scale, span): 適用したアフィンは物理 mm への逆算に要るので、
    呼び出し側でログに残せるように返す。
    """
    ref = ref_series.dropna()
    if len(ref) == 0:
        return processed_df, 0.0, 1.0, 0.0
    lo, hi = get_robust_min_max(ref)
    center = (lo + hi) / 2
    span = hi - lo
    scale = target_span / span if (span > 1e-9 and target_span > 0) else 1.0
    for col in cols:
        processed_df[col] = (processed_df[col].astype(float) - center) * scale
    return processed_df, center, scale, span


def process_pose_df(df, area_size, z_offset, rescale_ref='ankle'):
    """
    Convert the predicted camera-frame metre coordinates into sim-compatible metre coordinates.
    - Swap Y and Z so _X/_Y are the floor plane and _Z is height.
    - Floor plane (_X, _Y): min-max rescale the ANKLE spread to `area_size`, centred on
      the origin ([-area_size/2, +area_size/2]); the same per-axis affine is applied to every joint.
      This makes the person roam a sim-like range regardless of the (unreliable) predicted scale.

      基準は足首（--rescale_ref ankle、既定）。--area_size は「人が動き回れる床面の大きさ」
      であって「腰の軌跡の大きさ」ではないので、エリアの端まで到達しうる関節＝足首を
      合わせるのが定義に対して正しい。旧実装は腰中点を合わせていたため、足が常にエリアの
      外にはみ出す座標系になっていた。処理済み 41 セッションの実測で、腰を 2.0 m に
      合わせたとき足首の範囲は中央値 2.25 m（X）/ 2.27 m（Y）あり、座標全体が約 1.13 倍に
      引き伸ばされていた（腰の真の移動範囲は 2.0 m ではなく約 1.78 m）。
      この過大化は MPJPE の X/Y 成分にそのまま乗る（Z はリスケールしないので影響なし）。
      hip/ankle の比は 41 セッションで 0.888±0.021（X）とほぼ一定なので、旧データの
      X/Y をおおむね 0.89 倍したものが新しい座標系に相当する。
      全関節を基準にしないのは、腕はエリア外に出てよい（床を踏まない）ため。実測でも
      全関節基準の係数はセッション間のばらつきが 5 倍大きい（std 0.10 vs 0.02）。
    - Height (_Z): keep true metres (no rescale). Flip camera-down to up and apply z_offset so the
      bbox floor (per-frame z1) has a median of 0: height = z_offset - camera_down. z_offset is the
      value computed by calc_offset (median of the per-frame max camera-down).
    NOTE: the old x/y offsets and the *160 pixel mapping are intentionally removed here.
    """
    print(f"[4/5] Converting Pose data to metres (area size = {area_size} m, "
          f"ref = {rescale_ref}, z_offset = {z_offset})...")
    try:
        processed_df = df.copy()
        coord_cols = [col for col in df.columns if col not in ['timestamp', 'color_filepath']]

        # Y<->Z swap: afterwards _X=horizontal, _Y=depth (floor plane), _Z=camera-down (height src)
        for col_name in coord_cols:
            if col_name.endswith('_Y'):
                z_col_name = col_name.replace('_Y', '_Z')
                if z_col_name in coord_cols:
                    temp_y = processed_df[col_name].copy()
                    processed_df[col_name] = processed_df[z_col_name]
                    processed_df[z_col_name] = temp_y

        x_cols = [c for c in coord_cols if c.endswith('_X')]
        y_cols = [c for c in coord_cols if c.endswith('_Y')]
        z_cols = [c for c in coord_cols if c.endswith('_Z')]

        # Floor plane: rescale each axis to area_size using the ankle spread as reference.
        ref_x = _floor_reference(processed_df, 'X', rescale_ref)
        ref_y = _floor_reference(processed_df, 'Y', rescale_ref)
        processed_df, cx, sx, span_x = _rescale_axis(processed_df, x_cols, ref_x, area_size)
        processed_df, cy, sy, span_y = _rescale_axis(processed_df, y_cols, ref_y, area_size)
        # 適用した係数はここにしか残らない。物理メートルへ戻すとき（報告する MPJPE を
        # 実寸に換算するとき）に要るので、必ずログに出す。
        print(f"   └ Floor rescale (ref={rescale_ref}): "
              f"X span {span_x:.3f} m -> {area_size} m (scale {sx:.4f}, centre {cx:+.3f}), "
              f"Y span {span_y:.3f} m -> {area_size} m (scale {sy:.4f}, centre {cy:+.3f})")

        # Height: flip camera-down to up and shift by z_offset (calc_offset) so the floor median ~ 0.
        for col in z_cols:
            processed_df[col] = z_offset - processed_df[col].astype(float)

        print("Coordinate conversion complete (metres; floor plane rescaled to area size, height kept metric).")
        return processed_df
    except Exception as e:
        print(f"Error: An error occurred during Pose data processing: {e}")
        return None

def save_bbox_graph(bbox_df, check_dir, z_offset=None):
    """
    Plot the generated (metre) bbox coordinates across frames for verification.
    Panels: X (x1/x2), Y (y1/y2), Z after transform (z1/z2), Z before transform (camera_down).

    The Z-after panel uses `z1_raw` (the robust_min BEFORE --fix-z forces it to 0) when present,
    so the actual floor calibration is visible instead of a flat 0 line.
    The Z-before panel reconstructs the raw camera-down values via the exact inverse of the
    height transform (camera_down = z_offset - height) and marks z_offset, so you can check that
    the per-frame floor sits around z_offset (i.e. the z1 median lands on 0).
    Falls back to 3 panels when z1_raw / z_offset are unavailable.
    """
    try:
        os.makedirs(check_dir, exist_ok=True)
        df = bbox_df.sort_values('timestamp')
        idx = range(len(df))

        z1_after = df['z1_raw'] if 'z1_raw' in df.columns else df['z1']
        n_panels = 3

        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.4 * n_panels), sharex=True)
        axes[0].plot(idx, df['x1'], label='x1', color='blue')
        axes[0].plot(idx, df['x2'], label='x2', color='red')
        axes[0].set_ylabel('metre'); axes[0].set_title('bbox X (x1/x2)'); axes[0].legend()

        axes[1].plot(idx, df['y1'], label='y1', color='blue')
        axes[1].plot(idx, df['y2'], label='y2', color='red')
        axes[1].set_ylabel('metre'); axes[1].set_title('bbox Y (y1/y2)'); axes[1].legend()

        axes[2].plot(idx, z1_after, label='z1 (before fix-z)', color='green')
        axes[2].plot(idx, df['z2'], label='z2', color='orange')
        axes[2].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        axes[2].set_ylabel('metre'); axes[2].set_title('bbox Z after transform (height = z_offset - camera_down)')
        axes[2].legend()

        axes[-1].set_xlabel('frame')
        plt.tight_layout()
        out_path = os.path.join(check_dir, 'bbox_check.png')
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"└ bbox 座標グラフを保存しました（{n_panels}段）-> check/bbox_check.png")
    except Exception as e:
        print(f"Warning: failed to save bbox graph: {e}")


def get_robust_min_max(series, iqr_scale=1.5):
    """
    Returns the minimum and maximum values after removing outliers using the Interquartile Range (IQR).
    If there is too little data, or if everything is filtered out, returns the original min/max.
    """
    if len(series) < 4:
        return series.min(), series.max()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - (iqr * iqr_scale)
    upper_bound = q3 + (iqr * iqr_scale)
    
    filtered_series = series[(series >= lower_bound) & (series <= upper_bound)]
    
    if filtered_series.empty:
        return series.min(), series.max()
    
    return filtered_series.min(), filtered_series.max()

def generate_bounding_boxes_from_df(df, output_csv_path, num_people, fix_z=False, z_offset=None,
                                    bbox_size=BBOX_SIZE_M):
    """
    Generates 3D bounding box information (x,y,z) from the Pose data DataFrame and saves it as a CSV.
    Processes only frames where the number of detected people matches the specified number (num_people).
    z_offset is only used for the verification graph (to reconstruct the pre-transform camera-down z).
    """
    print(f"[5/5] Generating bounding boxes (Target: {num_people} people, size: {bbox_size} m) -> {os.path.basename(output_csv_path)}")
    try:
        bbox_data_list = []
        z1_raw_list = []  # robust_min before --fix-z forces z1 to 0 (graph only, not in the CSV)
        
        x_columns = [col for col in df.columns if col.endswith('_X')]
        y_columns = [col for col in df.columns if col.endswith('_Y')]
        z_columns = [col for col in df.columns if col.endswith('_Z')]
        required_cols = ['color_filepath', 'timestamp']
        required_cols += ['left_hip_X', 'right_hip_X', 'left_hip_Y', 'right_hip_Y']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"Error: Columns required for BBox generation {missing_cols} not found.")
            return False

        if not z_columns or not x_columns or not y_columns:
            print("Error: Some coordinate columns (_X, _Y, _Z) are missing.")
            return False

        print(f"└ [Step 1/3] Filtering number of people per frame (Target: {num_people} people)...")
        frame_person_counts = df.groupby('timestamp').transform('size')
        
        filtered_df = df[frame_person_counts == num_people].copy()

        original_frames = df['timestamp'].nunique()
        filtered_frames = filtered_df['timestamp'].nunique()
        original_rows = len(df)
        filtered_rows = len(filtered_df)

        if filtered_rows == 0:
            print(f"Error: 0 frames found where {num_people} people were detected. Aborting processing.")
            print(f"(Total frames: {original_frames}, Total detected rows: {original_rows})")
            return False
        
        print(f"Filtering complete.")
        print(f"Total frames: {original_frames} -> {filtered_frames} frames")
        print(f"Total detected rows: {original_rows} -> {filtered_rows} rows")

        print("└ [Step 2/3] Calculating BBox coordinates...")
        for index, row in filtered_df.iterrows():
            z_values = row[z_columns]
            z1, z2 = get_robust_min_max(z_values)
            z1_raw_list.append(z1)  # keep the measured floor for the verification graph
            if fix_z:
                z1 = 0.0
            
            center_x = (row['left_hip_X'] + row['right_hip_X']) / 2
            center_y = (row['left_hip_Y'] + row['right_hip_Y']) / 2

            # 水平は固定サイズ(metre)、hip 中心。center は正規化せずメートルのまま（sim と同じ）。
            half_size = bbox_size / 2
            x1, y1 = center_x - half_size, center_y - half_size
            x2, y2 = center_x + half_size, center_y + half_size

            bbox_info = {
                'timestamp': row['timestamp'], 'label': 0,
                'x1': x1, 'y1': y1, 'z1': z1,
                'x2': x2, 'y2': y2, 'z2': z2,
                'confidence': 1.0,
                'image_path': row['color_filepath'],
                'center_x': center_x,
                'center_y': center_y,
            }

            if 'person_id' in row:
                bbox_info['person_id'] = row['person_id']

            bbox_data_list.append(bbox_info)
        print("BBox coordinate calculation complete.")

        df_output = pd.DataFrame(bbox_data_list)

        df_output.to_csv(output_csv_path, index=False, float_format='%.6f')
        print("3D bounding box (metre) generation complete.")

        # Graph only: attach the pre-fix-z floor AFTER the CSV is written so the schema is unchanged.
        df_graph = df_output.copy()
        df_graph['z1_raw'] = z1_raw_list
        check_dir = os.path.join(os.path.dirname(output_csv_path), 'check')
        save_bbox_graph(df_graph, check_dir, z_offset=z_offset)
        return True
    except Exception as e:
        print(f"Error: An error occurred during BBox generation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Performs a series of processes from the original data (light_origin.csv, pose_origin.csv) to generate final products (light.csv, pose.csv, bbox_label.csv).\nA timestamp column is required in the input files.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    req_group = parser.add_argument_group('Required Arguments')
    trans_group = parser.add_argument_group('Transformation Options')

    req_group.add_argument("--data", type=str, required=True, help="Directory name containing the data to be processed.\nExample: my_animation")
    req_group.add_argument(
        "--shift",
        type=parse_hhmmss_to_seconds,
        required=True,
        metavar="hhmmss.ff",
        help="Time to adjust the timestamps in light.csv.\n"
             "Format: hhmmss.ff (Hours, Minutes, Seconds.Fractional seconds)\n"
             "Example: '000002.5' (Advance by 2.5 seconds)\n"
             "         '-000001.25' (Go back 1.25 seconds)"
    )
    req_group.add_argument(
        "--people",
        type=int,
        required=True,
        help="Target number of people (Required).\nFrames where the detected number of people does not match this number will be excluded from bbox_label.csv."
    )
    req_group.add_argument(
        "--area_size",
        type=float,
        required=True,
        metavar="METRES",
        help="Known size (metres) of the FLOOR AREA the person roams. The ankle spread is rescaled\n"
             "to this size and centred on the origin ([-area_size/2, +area_size/2]), so the pose spans\n"
             "a sim-like metre range regardless of the predicted scale.\n"
             "Change the anchor with --rescale_ref (ankle / hip)."
    )
    trans_group.add_argument(
        "--rescale_ref",
        type=str,
        default="ankle",
        choices=["ankle", "hip"],
        help="Reference used to fit the floor plane to --area_size.\n"
             "ankle (default): the left/right ankle spread, i.e. the floor the person actually\n"
             "  stepped on. This matches what --area_size means.\n"
             "hip: the hip-centre trajectory (the old behaviour). Only for reproducing results\n"
             "  produced before the fix; it inflates every X/Y coordinate by about 1.13x."
    )
    trans_group.add_argument(
        "--z_offset",
        type=float,
        default=1.0,
        help="Floor reference (metres) applied to height: height = z_offset - camera_down.\n"
             "Computed by calc_offset (median of per-frame max), so the bbox floor z1 median is ~0."
    )
    trans_group.add_argument(
        "--bbox_size",
        type=float,
        default=BBOX_SIZE_M,
        metavar="METRES",
        help=f"Horizontal (X/Y) side length of the bbox in metres, centred on the hip (default: {BBOX_SIZE_M}).\n"
             "Height (Z) is still taken from the joint Z robust min/max."
    )
    trans_group.add_argument(
        "--fix-z",
        action="store_true",
        help="Force z1=0.0 (fix bbox floor to ground level). Use for final run after offset verification."
    )
    trans_group.add_argument(
        "--smooth_median_window",
        type=int,
        default=SMOOTH_MEDIAN_WINDOW,
        help=f"Window (frames) of the rolling median that removes depth spikes (default: {SMOOTH_MEDIAN_WINDOW}).\n"
             "1 or less disables the median step."
    )
    trans_group.add_argument(
        "--smooth_mean_window",
        type=int,
        default=SMOOTH_MEAN_WINDOW,
        help=f"Window (frames) of the rolling mean applied after the median (default: {SMOOTH_MEAN_WINDOW}).\n"
             "1 or less disables the mean step."
    )
    trans_group.add_argument(
        "--smooth_max_gap",
        type=float,
        default=SMOOTH_MAX_GAP_SEC,
        metavar="SECONDS",
        help=f"Missing joints are interpolated only when the hole is at most this long (default: {SMOOTH_MAX_GAP_SEC} s,\n"
             "about 10 dropped frames at 30 fps). Longer holes stay missing and the row is dropped by\n"
             f"the joint filter. Segment splitting uses a separate threshold ({SMOOTH_SEGMENT_GAP_SEC} s), so raising\n"
             "this does not make smoothing average across a tracking dropout."
    )
    trans_group.add_argument(
        "--no-smooth",
        action="store_true",
        help="Skip interpolation and smoothing entirely (use the raw pose_origin values). For comparison runs."
    )
    trans_group.add_argument(
        "--joint_dist_tol",
        type=float,
        default=JOINT_DIST_TOL,
        help=f"Bone length (adjacent joint distance) check, LONG side (default: {JOINT_DIST_TOL}).\n"
             "A joint whose bone exceeds the person's own median by more than this fraction\n"
             f"(or {JOINT_DIST_SLACK_M} m, whichever is larger) is set to NaN, so a joint that picked up\n"
             "the background depth is removed instead of becoming training ground truth.\n"
             "Applied before smoothing, so short holes are refilled by interpolation."
    )
    trans_group.add_argument(
        "--joint_dist_short_tol",
        type=float,
        default=JOINT_DIST_SHORT_TOL,
        metavar="FRACTION",
        help="Bone length check, SHORT side (default: disabled).\n"
             "OFF BY DEFAULT ON PURPOSE: with a single depth camera a bone pointing along the viewing\n"
             "ray collapses to near zero, and which bones point that way depends on the action, so this\n"
             "side removes whole action classes rather than bad joints (measured on oneperson_gym_01:\n"
             "-35%% catches 97.9%% of the crouch phase). Check the per-phase removal rate before using it."
    )
    trans_group.add_argument(
        "--no-joint_dist",
        action="store_true",
        help="Skip the bone length filter entirely (keep every joint detect_yolo produced). For comparison runs."
    )

    phase_group = parser.add_argument_group("Action cycle / frame exclusion")
    phase_group.add_argument(
        "--cycle_phases",
        type=int,
        default=DEFAULT_CYCLE_PHASES,
        help=f"Number of phases in one action cycle (default: {DEFAULT_CYCLE_PHASES}).\n"
             "gym is 5 (walk, walk, stop, crouch, hand); meeting is 6 (+ bow)."
    )
    phase_group.add_argument(
        "--phase_sec",
        type=float,
        default=PHASE_DURATION_SEC,
        help=f"Duration of one phase in seconds; decimals allowed (default: {PHASE_DURATION_SEC}).\n"
             "Must match the value given to relabel.py."
    )
    phase_group.add_argument(
        "--drop_phases",
        default=None,
        metavar="JSON",
        help="JSON list of phases to remove entirely, e.g. '[5]' for the meeting bow.\n"
             "A frame is dropped when ANY person is in one of these phases."
    )
    phase_group.add_argument(
        "--start_img",
        default=None,
        help="Colour image that marks the crouch start. Its timestamp is the phase time origin.\n"
             "Required when --drop_phases is given."
    )
    phase_group.add_argument(
        "--shifts",
        default=None,
        help='Phase correction for a skipped cycle, as JSON:\n'
             '\'[{"from": "20260129_084644_593751.jpg", "delta": 1}]\'.\n'
             'Advances the phase by delta from that timestamp on (delta defaults to 1).\n'
             'Per-person drift goes in the "shifts" key of --persons instead.\n'
             'Must match the value passed to relabel.py.'
    )
    phase_group.add_argument(
        "--persons",
        default=None,
        metavar="JSON",
        help='Multi-person phase mapping: [{"ids": [...], "start_phase": int}, ...].\n'
             "Same value as relabel.py. Omit for single-person data."
    )

    args = parser.parse_args()

    persons = parse_persons_arg(args.persons, "[preprocess]")
    drop_phases = parse_drop_phases_arg(args.drop_phases, "[preprocess]")
    shifts = parse_shifts_arg(args.shifts, "[preprocess]")
    validate_cycle(args.cycle_phases, drop_phases, "[preprocess]", args.phase_sec)
    if drop_phases and not args.start_img:
        sys.exit("[preprocess] [error] --drop_phases を使うには --start_img が必要です")

    raw_path = raw_dir(args.data)
    data_path = processed_dir(args.data)

    # setup_data.py が raw から正規化コピーしたものを読む
    light_origin_path = os.path.join(data_path, 'light_origin.csv')
    timestamp_path = os.path.join(data_path, 'timestamp.csv')
    pose_origin_path = os.path.join(data_path, 'pose_origin.csv')
    # 正規化不要なので raw から直接読む
    environment_origin_path = os.path.join(raw_path, 'environment_origin.csv')

    light_final_path = os.path.join(data_path, 'light.csv')
    pose_final_path = os.path.join(data_path, 'pose.csv')
    environment_final_path = os.path.join(data_path, 'environment.csv')
    bbox_final_path = os.path.join(data_path, 'bbox_label.csv')

    print("=================================================")
    print(f"Starting processing. Target directory: {data_path}")
    print(f"Settings: Time Shift={args.shift:+.6f} sec, Target People={args.people}")
    print(f"   Movement area size: {args.area_size} m (pose in metres, centred origin, "
          f"ref={args.rescale_ref})")
    print("=================================================")

    try:
        print("\n--- Light Data Processing ---")
        light_df = load_light_with_ch_header(light_origin_path)
        if light_df is None: sys.exit("Failed to process Light data.")
        if args.shift != 0.0:
            light_df = shift_light_timestamp(light_df, args.shift)
            if light_df is None:
                sys.exit("Failed to modify Light data timestamps.")
        light_df.to_csv(light_final_path, index=False)
        print(f"[3/3] Saved final Light data -> {os.path.basename(light_final_path)}")

        if os.path.exists(environment_origin_path):
            print("\n--- Environment Data Processing ---")
            environment_df = load_light_with_ch_header(environment_origin_path)
            if environment_df is None: sys.exit("Failed to process Environment data.")
            if args.shift != 0.0:
                environment_df = shift_light_timestamp(environment_df, args.shift)
                if environment_df is None:
                    sys.exit("Failed to modify Environment data timestamps.")
            environment_df.to_csv(environment_final_path, index=False)
            print(f"[3/3] Saved final Environment data -> {os.path.basename(environment_final_path)}")
        else:
            print(f"\n--- Environment Data Processing --- [skip] environment_origin.csv が無いためスキップ")

        print("\n--- Pose Data Processing ---")
        print(f"[1/5] Starting original data read: {os.path.basename(pose_origin_path)}")
        pose_df = pd.read_csv(pose_origin_path)
        print(f"Original data read complete.")

        # 平滑化より前に掛ける。外れ値が rolling mean で近傍に拡散するのを防ぎ、
        # ここで空けた穴は直後の補間（0.38 秒以内）が埋め直す。
        if args.no_joint_dist:
            print("   └ [Additional Step] Skipping the bone length filter (--no-joint_dist).")
        else:
            pose_df = filter_joints_by_bone_length(pose_df, tol=args.joint_dist_tol,
                                                   short_tol=args.joint_dist_short_tol)

        if args.no_smooth:
            print("[2/5] Skipping Pose smoothing (--no-smooth).")
        else:
            pose_df = smooth_pose_df(pose_df,
                                     median_window=args.smooth_median_window,
                                     mean_window=args.smooth_mean_window,
                                     max_gap_sec=args.smooth_max_gap)

        print(f"[3/5] Reading Image Path data: {os.path.basename(timestamp_path)}")
        timestamp_df = pd.read_csv(timestamp_path)
        merge_cols = ['timestamp', 'color_filepath']
        if 'person_id' in timestamp_df.columns:
            merge_cols.append('person_id')
            print("person_id column detected. Adding to bbox_label.csv.")
        
        pose_df = pd.merge(pose_df, timestamp_df[merge_cols], on='timestamp', how='left')
        
        if pose_df['color_filepath'].isnull().any():
            print("Error: color_filepath matching the timestamp in pose_origin.csv was not found in timestamp.csv.")
            sys.exit("Failed to process Pose data.")
        print("Image Path data merge complete.")

        # process_pose_df より前に落とす（_rescale_axis に破棄対象を混ぜないため）
        pose_df = drop_phase_frames(pose_df, persons, args.start_img,
                                    args.cycle_phases, drop_phases, shifts, args.phase_sec)

        pose_df = process_pose_df(pose_df, args.area_size, args.z_offset, args.rescale_ref)
        if pose_df is None: sys.exit("Failed to process Pose data.")

        pose_df = remove_duplicate_pose_frames(pose_df, threshold=DUPLICATE_THRESHOLD_M)

        # 必須関節は学習側 datasets/dataset.py に合わせる（実測 12 点 + 頭部代表 1 点）。
        # dataset.py はこれらが欠けた行を無言で dropna するため、ここで落としておかないと
        # 後続の人数フィルタを通ったフレームが学習時に人数不足の教師に化ける。
        #
        # 頭部は鼻 1 点ではなく顔 5 点（鼻・両目・両耳）の「いずれか 1 点以上」を条件にする。
        # 実測では個々の取得率が 87〜93% なのに対し、いずれか 1 点以上なら 99.8%。
        # 向きによって落ちる点が違う（後ろ向きで鼻が消えても耳は残る）ため。
        # dataset.py 側は取れている点の平均を head として合成する。
        print(f"   └ [Additional Step] Filtering body joints "
              f"(Face 1/{len(keypoints.FACE_JOINTS)} + Hips required + Others: {10}/10)...")
        hip_joints = ['left_hip', 'right_hip']
        other_body_parts = ['shoulder', 'elbow', 'wrist', 'knee', 'ankle']
        other_body_x_cols = [
            col for col in pose_df.columns 
            if any(part in col.lower() for part in other_body_parts) and col.endswith('_X')
        ]
        required_cols = [c for j in keypoints.FACE_JOINTS + hip_joints
                         for c in keypoints.joint_cols(j)]
        missing_required = [c for c in required_cols if c not in pose_df.columns]
        if missing_required:
            print(f"Error: Columns required for the joint filter {missing_required} not found.")
            sys.exit("Failed to process Pose data.")

        def is_valid(df_subset):
            return df_subset.notna() & (df_subset != 0)

        initial_rows = len(pose_df)
        # head を作れるか（顔 5 点のうち 1 点以上）。判定は学習側と同じ実装を使う。
        has_face = keypoints.face_available_mask(pose_df)
        has_both_hips = (keypoints.joint_available(pose_df, 'left_hip')
                         & keypoints.joint_available(pose_df, 'right_hip'))
        other_valid_count = is_valid(pose_df[other_body_x_cols]).sum(axis=1)
        meets_threshold = other_valid_count >= 10
        pose_df = pose_df[has_face & has_both_hips & meets_threshold].copy()
        
        removed_rows = initial_rows - len(pose_df)
        if removed_rows > 0:
            # 重複するケースは先の条件に計上される（内訳の合計 = removed_rows）
            face_missing = int((~has_face).sum())
            hip_missing = int((has_face & ~has_both_hips).sum())
            joint_missing = removed_rows - face_missing - hip_missing
            print(f"Exclusion complete: Removed a total of {removed_rows} rows.")
            print(f"(No face joint: {face_missing} / Missing hips: {hip_missing} / "
                  f"Insufficient other joints: {joint_missing})")
        else:
            print("All rows meet the conditions (Face + Hips required + Joint count).")
        
        print(f"   └ Filtering number of people in Pose data (Target: {args.people} people)...")
        counts = pose_df.groupby('timestamp').transform('size')
        pose_df = pose_df[counts == args.people].copy()
        
        pose_df.to_csv(pose_final_path, index=False, float_format='%.6f')
        print(f"Saved final Pose data -> {os.path.basename(pose_final_path)}")

        if not generate_bounding_boxes_from_df(pose_df, bbox_final_path, args.people,
                                               fix_z=args.fix_z, z_offset=args.z_offset,
                                               bbox_size=args.bbox_size):
            sys.exit("Failed to process Pose data.")

    except FileNotFoundError as e:
        print(f"Fatal Error: Input file not found. Please check if '{e.filename}' exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal Error: Processing aborted due to an unexpected error: {e}")
        sys.exit(1)

    print("\n=================================================")
    print("All processing completed successfully.")
    print("Final Output Files:")
    print(f"   - {light_final_path}")
    print(f"   - {pose_final_path}")
    if os.path.exists(environment_final_path):
        print(f"   - {environment_final_path}")
    print(f"   - {bbox_final_path}")
    print("=================================================")

if __name__ == '__main__':
    main()