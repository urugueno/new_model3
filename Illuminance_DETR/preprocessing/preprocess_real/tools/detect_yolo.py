import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_ROOT, YOLO_WEIGHTS, SAM3_WEIGHTS, raw_dir, processed_dir
from phase import filename_to_timestamp

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def deproject_pixel_to_point(intrinsics, x, y, depth):
    rs_intrinsics = rs.intrinsics()
    rs_intrinsics.width = intrinsics['width']
    rs_intrinsics.height = intrinsics['height']
    rs_intrinsics.ppx = intrinsics['ppx']
    rs_intrinsics.ppy = intrinsics['ppy']
    rs_intrinsics.fx = intrinsics['fx']
    rs_intrinsics.fy = intrinsics['fy']

    if intrinsics.get('model') == 'inverse_brown_conrady':
        rs_intrinsics.model = rs.distortion.inverse_brown_conrady
        rs_intrinsics.coeffs = intrinsics['coeffs']
    elif intrinsics.get('model') == 'brown_conrady' or intrinsics.get('model') is None:
         rs_intrinsics.model = rs.distortion.brown_conrady
         rs_intrinsics.coeffs = intrinsics['coeffs']
    else:
        rs_intrinsics.model = rs.distortion.none

    point = rs.rs2_deproject_pixel_to_point(rs_intrinsics, [x, y], depth)
    return point

def pose_row(timestamp, person_id, keypoints, confs, image, depth_array, intrinsics):
    """1 人ぶんの 2D キーポイントを深度で 3D に戻し、pose_origin.csv の 1 行にする。"""
    frame_pose_data = {'timestamp': timestamp, 'person_id': person_id}
    h, w, _ = image.shape
    for kpt_idx, kpt_name in enumerate(COCO_KEYPOINT_NAMES):
        cx, cy = int(keypoints[kpt_idx][0]), int(keypoints[kpt_idx][1])
        conf = confs[kpt_idx]

        if 0 <= cx < w and 0 <= cy < h and conf >= 0.2:
            depth_in_mm = depth_array[cy, cx]
            if depth_in_mm > 0:
                point_3d = deproject_pixel_to_point(intrinsics, cx, cy, depth_in_mm * 0.001)
                frame_pose_data[f'{kpt_name}_X'] = point_3d[0]
                frame_pose_data[f'{kpt_name}_Y'] = point_3d[1]
                frame_pose_data[f'{kpt_name}_Z'] = point_3d[2]
                continue

        frame_pose_data[f'{kpt_name}_X'] = None
        frame_pose_data[f'{kpt_name}_Y'] = None
        frame_pose_data[f'{kpt_name}_Z'] = None
    return frame_pose_data

def empty_row(timestamp):
    frame_pose_data = {'timestamp': timestamp, 'person_id': None}
    for kpt_name in COCO_KEYPOINT_NAMES:
        frame_pose_data[f'{kpt_name}_X'] = frame_pose_data[f'{kpt_name}_Y'] = frame_pose_data[f'{kpt_name}_Z'] = None
    return frame_pose_data

def box_iou(a, b):
    """a: (N,4), b: (M,4) の xyxy 同士の IoU 行列。"""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)

def run_bytetrack(yolo_model, frames, intrinsics):
    all_pose_data = []
    for row, color_filepath, depth_filepath in tqdm(frames):
        image = cv2.imread(color_filepath)
        depth_array = np.load(depth_filepath)

        if image is None:
            continue

        results = yolo_model.track(image, persist=True, tracker="bytetrack.yaml", verbose=False)

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints_all = results[0].keypoints.xy.cpu().numpy()
            confs_all = results[0].keypoints.conf.cpu().numpy()
            for i, track_id in enumerate(track_ids):
                all_pose_data.append(pose_row(row['timestamp'], track_id, keypoints_all[i], confs_all[i],
                                              image, depth_array, intrinsics))
        else:
            all_pose_data.append(empty_row(row['timestamp']))
    return all_pose_data

def run_sam3(yolo_model, frames, intrinsics, video_path, sam3_weights, sam3_imgsz, iou_thresh, min_frames):
    """SAM3（テキスト "person"）で人物 ID を追跡し、各フレームの YOLO-pose の検出を IoU で ID に割り当てる。

    SAM3 は動画入力しか受け付けないので、対象フレームを一時 mp4 にしてから流す。
    SAM3 の短命な track（人が重なった時に腕などを別人として拾ったもの）は min_frames 未満なら捨てる。
    """
    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    h, w = None, None
    for _, color_filepath, _ in frames:
        img = cv2.imread(color_filepath)
        if img is not None:
            h, w = img.shape[:2]
            break
    vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for _, color_filepath, _ in tqdm(frames, desc="sam3 input mp4"):
        img = cv2.imread(color_filepath)
        # 読めない画像も黒フレームで埋めて、mp4 のフレーム番号と frames の添字を揃える
        vw.write(img if img is not None else np.zeros((h, w, 3), np.uint8))
    vw.release()

    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=sam3_imgsz, model=sam3_weights,
                     half=True, verbose=False, save=False)
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    sam3_stream = predictor(source=video_path, text=["person"], stream=True)

    all_pose_data = []
    try:
        for (row, color_filepath, depth_filepath), sam3_result in tqdm(zip(frames, sam3_stream), total=len(frames)):
            image = cv2.imread(color_filepath)
            if image is None:
                continue
            depth_array = np.load(depth_filepath)

            rows = []
            sam3_boxes = sam3_result.boxes
            if sam3_boxes is not None and sam3_boxes.id is not None and len(sam3_boxes):
                pose = yolo_model.predict(image, classes=[0], verbose=False)[0]
                if len(pose.boxes):
                    iou = box_iou(sam3_boxes.xyxy.cpu().numpy(), pose.boxes.xyxy.cpu().numpy())
                    # SAM3 の id は 0 始まりなので ByteTrack と同じく 1 始まりにそろえる
                    sam3_ids = sam3_boxes.id.int().cpu().numpy() + 1
                    keypoints_all = pose.keypoints.xy.cpu().numpy()
                    confs_all = pose.keypoints.conf.cpu().numpy()
                    for si, pi in zip(*linear_sum_assignment(-iou)):
                        if iou[si, pi] >= iou_thresh:
                            rows.append(pose_row(row['timestamp'], int(sam3_ids[si]), keypoints_all[pi],
                                                 confs_all[pi], image, depth_array, intrinsics))
            all_pose_data += rows or [empty_row(row['timestamp'])]
    finally:
        os.remove(video_path)

    df = pd.DataFrame(all_pose_data)
    counts = df['person_id'].value_counts()
    short_ids = counts[counts < min_frames].index
    if len(short_ids):
        print(f"[sam3] {min_frames} フレーム未満の track を削除: "
              + ", ".join(f"{int(i)}({counts[i]})" for i in short_ids))
        df = df[~df['person_id'].isin(short_ids)]
        # 誰も残らなかったフレームは「未検出」の行で埋め直す（bytetrack 時と同じ形にそろえる）
        missing = sorted(set(r['timestamp'] for r in all_pose_data) - set(df['timestamp']))
        df = pd.concat([df, pd.DataFrame([empty_row(t) for t in missing])]).sort_values('timestamp', kind='stable')
    print(f"[sam3] 残った person_id: {sorted(int(i) for i in df['person_id'].dropna().unique())}")
    return df.to_dict('records')

def main():
    parser = argparse.ArgumentParser(
        description="Estimate 3D pose from recorded RealSense data using YOLOv8. "
                    "pose_origin.csv holds the RAW per-frame estimates (undetected joints stay empty); "
                    "gap interpolation and smoothing are done later in preprocess_real.py."
    )
    parser.add_argument("--data", type=str, required=True, help="The timestamp directory of the data.")
    parser.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "sam3"],
                        help="person_id の付け方。bytetrack: YOLO+ByteTrack（従来）/ "
                             "sam3: SAM3 で追跡し YOLO-pose の検出を IoU で対応づける（ID が途切れにくいが約 7 fps）")
    parser.add_argument("--start_img", default=None,
                        help="この画像より前のフレームは処理しない（どうせ trim_pose で消える）。"
                             "sam3 では全員がそろったフレームから始めると ID が安定する")
    parser.add_argument("--sam3_weights", default=SAM3_WEIGHTS)
    parser.add_argument("--sam3_imgsz", type=int, default=644)
    parser.add_argument("--sam3_iou", type=float, default=0.3, help="SAM3 と YOLO の box を対応づける IoU 下限")
    parser.add_argument("--sam3_min_frames", type=int, default=90,
                        help="これ未満のフレーム数しか続かない SAM3 の track は誤検出として捨てる")
    args = parser.parse_args()

    FILE_NAME = args.data
    src_dir = raw_dir(FILE_NAME)
    dst_dir = processed_dir(FILE_NAME)
    INPUT_CSV = os.path.join(dst_dir, 'timestamp.csv')
    OUTPUT_CSV = os.path.join(dst_dir, 'pose_origin.csv')
    INTRINSICS_FILE = os.path.join(src_dir, 'intrinsics.json')

    if not os.path.exists(INPUT_CSV):
        print(f"Error: timestamp.csv not found: {INPUT_CSV}（先に setup_data.py を実行してください）")
        return
    if not os.path.exists(INTRINSICS_FILE):
        print(f"Error: intrinsics.json not found: {INTRINSICS_FILE}")
        return

    print("Loading YOLOv8x-pose model...")
    yolo_model = YOLO(YOLO_WEIGHTS)

    with open(INTRINSICS_FILE) as f:
        intrinsics = json.load(f)

    df_timestamp = pd.read_csv(INPUT_CSV)
    if args.start_img:
        df_timestamp = df_timestamp[df_timestamp['timestamp'] >= filename_to_timestamp(args.start_img)]
    frames = [(row, os.path.join(RAW_ROOT, row['color_filepath']), os.path.join(RAW_ROOT, row['depth_filepath']))
              for _, row in df_timestamp.iterrows()]

    print(f"Processing frames (tracker={args.tracker})...")
    if args.tracker == "sam3":
        all_pose_data = run_sam3(yolo_model, frames, intrinsics, os.path.join(dst_dir, '_sam3_input.mp4'),
                                 args.sam3_weights, args.sam3_imgsz, args.sam3_iou, args.sam3_min_frames)
    else:
        all_pose_data = run_bytetrack(yolo_model, frames, intrinsics)

    if not all_pose_data:
        print("No poses detected.")
        return

    df_pose = pd.DataFrame(all_pose_data)

    # No filtering here on purpose: pose_origin.csv keeps the raw estimates so that undetected
    # joints stay missing. Interpolation / smoothing happens in preprocess_real.py, after the
    # outlier and joint-count filters.
    df_pose.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved (raw, unsmoothed) to: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
