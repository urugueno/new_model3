"""
学習で使う関節の定義。前処理・検証動画・学習コードで共有する唯一の定義元。

pose_origin.csv には COCO 17 点が入っているが、学習で使うのはこのうち 12 点
（BODY_JOINTS）と、そこから合成する 2 点（neck, head）の計 14 点だけ。

頭部を鼻 1 点で代表させると、後ろ向き・下向きの姿勢で大量に脱落する
（実測の取得率 92.8%）。顔 5 点（FACE_JOINTS）のうち「取れているものの平均」を
head とすれば 99.8% 取れる（向きによって落ちる点が違い、後ろ向きでも耳が残るため）。

この定義がズレると、前処理が通した行を学習側が無言で捨てたり、検証動画が
学習と違う骨格を描いたりするので、必ずここを参照すること。
"""
import numpy as np
import pandas as pd

AXES = ("X", "Y", "Z")

# pose.csv からそのまま読む実測 12 点。すべて揃っている行だけを使う。
BODY_JOINTS = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 頭部代表点 head の材料。このうち 1 点以上あれば head を作れる。
FACE_JOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]

# モデルが入出力する 14 点（実測 12 + 合成 2）
OUTPUT_JOINTS = BODY_JOINTS + ["neck", "head"]

# 描画用の骨格。OUTPUT_JOINTS だけで構成する（目・耳・鼻は個別には描かない）。
SKELETON = [
    # 頭・首
    ("head", "neck"),
    ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    # 体幹
    ("left_shoulder", "right_shoulder"), ("right_shoulder", "right_hip"),
    ("right_hip", "left_hip"), ("left_hip", "left_shoulder"),
    # 腕
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    # 脚
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

# 骨長チェック用の隣接関節ペア。pose_origin.csv の COCO 17 点空間で定義する。
# SKELETON とは用途が違うので別に持つ:
#   SKELETON     … 描画用。合成点 neck/head を含むので生データには存在しない列を指す。
#   RAW_SKELETON … 検証用。pose_origin.csv に実在する 17 点だけで構成する。
# 顔↔肩の 2 本を入れているのは、顔 5 点がまとめて背景の depth に飛んだ場合、
# 顔の内部（鼻-目・目-耳）の骨長は正常なままで、体幹との接続でしか異常を検出できないため。
RAW_SKELETON = [
    # 顔
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    # 顔と体幹をつなぐ
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"),
    # 体幹
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    # 腕
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    # 脚
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

# RAW_SKELETON の各骨を「左右をまとめた論理骨」に写す。基準長はこの単位で取る。
# 左右対称な骨は同じ長さのはずなので、まとめるとサンプル数が倍になり、
# 片側だけ慢性的にズレているケース（利き手側が壁際に来続けた等）にも引きずられにくい。
SYMMETRIC_BONES = {
    ("nose", "left_eye"): "nose_eye",
    ("nose", "right_eye"): "nose_eye",
    ("left_eye", "left_ear"): "eye_ear",
    ("right_eye", "right_ear"): "eye_ear",
    ("left_ear", "left_shoulder"): "ear_shoulder",
    ("right_ear", "right_shoulder"): "ear_shoulder",
    ("left_shoulder", "right_shoulder"): "shoulder_width",
    ("left_hip", "right_hip"): "hip_width",
    ("left_shoulder", "left_hip"): "torso_side",
    ("right_shoulder", "right_hip"): "torso_side",
    ("left_shoulder", "left_elbow"): "upper_arm",
    ("right_shoulder", "right_elbow"): "upper_arm",
    ("left_elbow", "left_wrist"): "forearm",
    ("right_elbow", "right_wrist"): "forearm",
    ("left_hip", "left_knee"): "thigh",
    ("right_hip", "right_knee"): "thigh",
    ("left_knee", "left_ankle"): "shank",
    ("right_knee", "right_ankle"): "shank",
}

# 骨長違反の「犯人」を選ぶときの優先順位。値が大きいほど先に捨ててよい。
# 体幹に近い関節ほど検出が安定していて、かつ落とすと後段の必須条件
# （両腰 + 他 10 関節）を一発で壊すので小さくしてある。逆に手首・足首・顔は
# 動きが速く depth を取り違えやすいうえ、顔は 5 点のうち 1 点残ればよい。
JOINT_DROP_PRIORITY = {
    "left_hip": 0, "right_hip": 0,
    "left_shoulder": 1, "right_shoulder": 1,
    "left_knee": 2, "right_knee": 2,
    "left_elbow": 3, "right_elbow": 3,
    "left_ankle": 4, "right_ankle": 4,
    "left_wrist": 5, "right_wrist": 5,
    "nose": 6, "left_eye": 6, "right_eye": 6, "left_ear": 6, "right_ear": 6,
}


def joint_cols(joint):
    """関節名から X/Y/Z の列名を返す。"""
    return [f"{joint}_{ax}" for ax in AXES]


def joint_available(df, joint):
    """X/Y/Z が 3 つとも揃っている行を True で返す。列が無ければ全 False。"""
    cols = joint_cols(joint)
    if not all(c in df.columns for c in cols):
        return pd.Series(False, index=df.index)
    return df[cols].notna().all(axis=1)


def bone_length(df, joint_a, joint_b):
    """
    2 関節間のユークリッド距離を行ごとに返す。

    どちらかの関節の X/Y/Z が 1 つでも欠けている行は NaN（評価不能）。列自体が無い場合も
    全 NaN を返す（joint_available が全 False を返すのと同じ扱い）。

    座標がメートルであることが前提。preprocess_real.py の process_pose_df は X/Y を軸ごとに
    別倍率でリスケールするため、それより後の座標で呼んでも長さに意味は無い。
    """
    cols_a, cols_b = joint_cols(joint_a), joint_cols(joint_b)
    if not all(c in df.columns for c in cols_a + cols_b):
        return pd.Series(np.nan, index=df.index)
    diff = df[cols_a].to_numpy(dtype=float) - df[cols_b].to_numpy(dtype=float)
    return pd.Series(np.sqrt((diff ** 2).sum(axis=1)), index=df.index)


def face_available_mask(df):
    """顔 5 点のうち 1 点以上使える行を True で返す（head を作れる行）。"""
    mask = pd.Series(False, index=df.index)
    for joint in FACE_JOINTS:
        mask |= joint_available(df, joint)
    return mask


def head_point(row):
    """
    取れている顔関節の平均を頭部代表点 (x, y, z) として返す。
    1 点も取れていなければ None（そういう行は face_available_mask で除外しておく）。
    """
    pts = []
    for joint in FACE_JOINTS:
        xyz = [row.get(c) for c in joint_cols(joint)]
        if all(v is not None and not pd.isna(v) for v in xyz):
            pts.append([float(v) for v in xyz])
    if not pts:
        return None
    return np.mean(pts, axis=0)


def neck_point(row):
    """両肩の中点を首として返す。肩が欠けていれば None。"""
    pts = []
    for joint in ("left_shoulder", "right_shoulder"):
        xyz = [row.get(c) for c in joint_cols(joint)]
        if not all(v is not None and not pd.isna(v) for v in xyz):
            return None
        pts.append([float(v) for v in xyz])
    return np.mean(pts, axis=0)


def output_points(row):
    """
    学習に渡るのと同じ 14 点を {関節名: (x, y, z)} で返す。
    取れていない点は含めない（描画側でスキップできるように）。
    """
    points = {}
    for joint in BODY_JOINTS:
        xyz = [row.get(c) for c in joint_cols(joint)]
        if all(v is not None and not pd.isna(v) for v in xyz):
            points[joint] = tuple(float(v) for v in xyz)
    neck = neck_point(row)
    if neck is not None:
        points["neck"] = tuple(neck)
    head = head_point(row)
    if head is not None:
        points["head"] = tuple(head)
    return points
