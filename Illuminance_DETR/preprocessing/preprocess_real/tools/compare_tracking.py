"""YOLO+ByteTrack / YOLO+BoT-SORT(ReID) / SAM3 のトラッキングを GT なしの指標で簡易比較する。

指標（人数 --people が既知である前提）:
  n_ids        : 出現した track_id の総数（理想は people 人と同数）
  new_ids_after: 先頭フレーム以降に新規発生した id 数（= ID 分断・乗り移りの回数の目安）
  miss_frames  : 検出数 != people のフレーム割合
  mean_len     : track 1 本あたりの平均フレーム数
  fps          : 処理速度

使い方:
  CUDA_VISIBLE_DEVICES=1 python compare_tracking.py --data threeperson_gym_01 --people 3 \
      --start 0 --num_frames 1800 --out_dir /tmp/cmp
SAM3 の重み（facebook/sam3 の sam3.pt, 要アクセス申請）は --sam3_weights で渡す。
"""
import os
import sys
import time
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YOLO_WEIGHTS, raw_dir


def load_frames(data, start, num_frames, stride):
    df = pd.read_csv(os.path.join(raw_dir(data), "timestamp.csv"))
    df = df.iloc[start:start + num_frames * stride:stride].reset_index(drop=True)
    # raw の timestamp.csv は "data/<name>/color/xxx.jpg" 形式なので basename だけ使う
    return [os.path.join(raw_dir(data), "color", os.path.basename(p)) for p in df["color_filepath"]]


def write_video(paths, out_path, fps=30):
    h, w = cv2.imread(paths[0]).shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in tqdm(paths, desc="mp4"):
        vw.write(cv2.imread(p))
    vw.release()


def boxes_to_rows(method, frame_idx, boxes):
    if boxes is None or boxes.id is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().tolist()
    conf = boxes.conf.cpu().numpy()
    return [dict(method=method, frame=frame_idx, id=i, x1=b[0], y1=b[1], x2=b[2], y2=b[3], conf=c)
            for i, b, c in zip(ids, xyxy, conf)]


def run_yolo(paths, tracker, method):
    from ultralytics import YOLO
    model = YOLO(YOLO_WEIGHTS)
    rows = []
    t0 = time.time()
    for f, p in enumerate(tqdm(paths, desc=method)):
        r = model.track(cv2.imread(p), persist=True, tracker=tracker, classes=[0], verbose=False)
        rows += boxes_to_rows(method, f, r[0].boxes)
    return rows, len(paths) / (time.time() - t0)


def run_sam3(video_path, weights, n_frames, imgsz, half):
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=imgsz, model=weights, half=half,
                     verbose=False, save=False)
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    rows = []
    t0 = time.time()
    for f, r in enumerate(tqdm(predictor(source=video_path, text=["person"], stream=True),
                               total=n_frames, desc="sam3")):
        rows += boxes_to_rows("sam3", f, r.boxes)
    return rows, n_frames / (time.time() - t0)


def metrics(df, n_frames, people, fps):
    if df.empty:
        return dict(n_ids=0, new_ids_after=0, miss_frames=1.0, mean_len=0.0, fps=fps)
    first_ids = set(df[df.frame == df.frame.min()].id)
    counts = df.groupby("frame").size().reindex(range(n_frames), fill_value=0)
    return dict(
        n_ids=df.id.nunique(),
        new_ids_after=df.id.nunique() - len(first_ids),
        miss_frames=round(float((counts != people).mean()), 3),
        mean_len=round(float(df.groupby("id").size().mean()), 1),
        fps=round(fps, 1),
    )


def draw_compare(paths, df, methods, out_path, fps=30, scale=0.5):
    palette = [tuple(int(c) for c in cv2.cvtColor(np.uint8([[[(i * 37) % 180, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
               for i in range(64)]
    by = {m: {f: g for f, g in df[df.method == m].groupby("frame")} for m in methods}
    vw = None
    for f, p in enumerate(tqdm(paths, desc="video")):
        img = cv2.imread(p)
        tiles = []
        for m in methods:
            t = img.copy()
            g = by[m].get(f)
            if g is not None:
                for r in g.itertuples():
                    col = palette[r.id % len(palette)]
                    cv2.rectangle(t, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 3)
                    cv2.putText(t, str(r.id), (int(r.x1), int(r.y1) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
            cv2.putText(t, f"{m}  f={f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            tiles.append(cv2.resize(t, None, fx=scale, fy=scale))
        frame = np.hstack(tiles)
        if vw is None:
            vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.shape[1::-1])
        vw.write(frame)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--people", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=1800)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--methods", nargs="+", default=["bytetrack", "botsort_reid", "sam3"])
    ap.add_argument("--sam3_weights", default="sam3.pt")
    ap.add_argument("--sam3_imgsz", type=int, default=644)
    ap.add_argument("--no_half", action="store_true")
    ap.add_argument("--no_video", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = load_frames(args.data, args.start, args.num_frames, args.stride)
    n = len(paths)

    all_rows, summary = [], {}
    for m in args.methods:
        if m == "bytetrack":
            rows, fps = run_yolo(paths, "bytetrack.yaml", m)
        elif m == "botsort_reid":
            cfg = os.path.join(args.out_dir, "botsort_reid.yaml")
            with open(cfg, "w") as f:
                f.write("tracker_type: botsort\ntrack_high_thresh: 0.25\ntrack_low_thresh: 0.1\n"
                        "new_track_thresh: 0.25\ntrack_buffer: 60\nmatch_thresh: 0.8\nfuse_score: True\n"
                        "gmc_method: sparseOptFlow\nproximity_thresh: 0.5\nappearance_thresh: 0.8\n"
                        "with_reid: True\nmodel: auto\n")
            rows, fps = run_yolo(paths, cfg, m)
        elif m == "sam3":
            video = os.path.join(args.out_dir, "input.mp4")
            if not os.path.exists(video):
                write_video(paths, video)
            rows, fps = run_sam3(video, args.sam3_weights, n, args.sam3_imgsz, not args.no_half)
        else:
            raise ValueError(m)
        all_rows += rows
        summary[m] = metrics(pd.DataFrame(rows, columns=["method", "frame", "id", "x1", "y1", "x2", "y2", "conf"]),
                             n, args.people, fps)
        print(m, summary[m])

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out_dir, "tracks.csv"), index=False)
    table = pd.DataFrame(summary).T
    table.to_csv(os.path.join(args.out_dir, "summary.csv"))
    print(f"\n{args.data} frames {args.start}..{args.start + n * args.stride} (n={n}, people={args.people})")
    print(table.to_string())
    if not args.no_video:
        draw_compare(paths, df, args.methods, os.path.join(args.out_dir, "compare.mp4"))


if __name__ == "__main__":
    main()
