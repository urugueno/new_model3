import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import random

from util.calculate import box_xyzxyz_to_cxcywhd
from util import keypoints

SENSOR_COORDS_FILENAME = "sensor_coords.json"


def load_sensor_coords(ill_path, columns, selected_columns):
    """
    selected_columns の各センサの正規化座標 (N, 2) と、その正規化枠の名前を返す。

    実データは light.csv と同じフォルダの sensor_coords.json（列名 = センサ ID -> 座標）から引く。
    マットをずらした収録ではデータごとに座標が違うので、ファイルごとに読む必要がある
    （preprocessing/preprocess_real/sensor_layout.py 参照）。json が無いもの（シミュレーションと
    移行前の実データ）は列名 "x,y" をそのまま座標として読む旧形式。
    """
    json_path = os.path.join(os.path.dirname(ill_path), SENSOR_COORDS_FILENAME)
    if not selected_columns:
        # RGB / 音モダリティは照度列を使わない（main.py が selected_columns を [] にする）
        return torch.zeros((0, 2), dtype=torch.float32), None
    if os.path.exists(json_path):
        with open(json_path) as f:
            payload = json.load(f)
        coords_map = {ch: v["norm"] for ch, v in payload["sensors"].items()}
        frame = payload.get("frame_m", {}).get("name", "special")
    else:
        frame = "header"
        try:
            coords_map = {c: list(map(float, c.split(','))) for c in columns[1:]}
        except ValueError:
            raise ValueError(
                f"{json_path} が無く、列名も座標形式ではありません: {ill_path}。"
                f"preprocessing/preprocess_real/tools/sensor_coords.py で作成してください")
    missing = [c for c in selected_columns if c not in coords_map]
    if missing:
        raise ValueError(f"センサ {missing} の座標がありません: {ill_path}")
    coords = torch.tensor([coords_map[c] for c in selected_columns],
                          dtype=torch.float32).reshape(-1, 2)
    if torch.any(coords < 0) or torch.any(coords > 1):
        raise ValueError(f"Sensor coordinates are not normalized to [0, 1]: {ill_path}")
    return coords, frame


class IlluminanceDetectionDataset(Dataset):
    # 関節の定義は util/keypoints.py が唯一の定義元。前処理・検証動画と共有する。
    KEYPOINT_NAMES = keypoints.BODY_JOINTS          # 実測 12 点
    FACE_NAMES = keypoints.FACE_JOINTS              # head の材料（5 点中 1 点以上あれば可）
    OUTPUT_KEYPOINT_NAMES = keypoints.OUTPUT_JOINTS  # 実測 12 + 合成 2（neck, head）= 14

    _LEFT_SHOULDER = KEYPOINT_NAMES.index("left_shoulder")
    _RIGHT_SHOULDER = KEYPOINT_NAMES.index("right_shoulder")

    def __init__(self, illuminance_csv_path, bbox_csv_path, environment_csv_path, args, split='train', time_format='%Y-%m-%d %H:%M:%S %f', do_split=False):
        assert split in ['train', 'val', 'test']
        
        self.task = args.task

        is_list_input = isinstance(illuminance_csv_path, list)
        merged_df_split = None
        ill_dfs_all_concat = []
        self.environment_vectors_raw = []
        self.all_env_sensor_names = []
        current_ill_index_offset = 0

        if self.task == 'pose':
            required_tgt_cols = ['timestamp']
            for name in self.KEYPOINT_NAMES:
                required_tgt_cols.extend([f'{name}_X', f'{name}_Y', f'{name}_Z'])
        else:
            required_tgt_cols = ['timestamp', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'label']

        if not is_list_input:
            illuminance_csv_path = [illuminance_csv_path]
            bbox_csv_path = [bbox_csv_path]
            environment_csv_path = [environment_csv_path]

        merged_files_to_concat = []
        test_file_idx = 0
        # file_idx 順に (light.csv のパス, 列名)。座標は selected_columns が決まってから引く
        coord_sources = []

        for source_dataset_id, (ill_path, bbox_path, env_path) in enumerate(zip(illuminance_csv_path, bbox_csv_path, environment_csv_path)):
            try:
                ill_df_single = pd.read_csv(ill_path)
                bbox_df_single = pd.read_csv(bbox_path)
            except FileNotFoundError as e:
                print(f"ERROR: Could not d file {e.filename}. Skipping this file.")
                continue

            ill_df_single.dropna(inplace=True)
            
            check_subset = [c for c in required_tgt_cols if c in bbox_df_single.columns]
            bbox_df_single.dropna(subset=check_subset, inplace=True)

            if self.task == 'pose':
                # head は顔 5 点のうち「取れているもの」の平均なので、dropna（全列必須）では
                # 表現できない。1 点も取れていない行だけをここで落とす。
                n_before = len(bbox_df_single)
                bbox_df_single = bbox_df_single[
                    keypoints.face_available_mask(bbox_df_single)].copy()
                n_dropped = n_before - len(bbox_df_single)
                if n_dropped:
                    print(f"Dropped {n_dropped} rows with no usable face keypoint "
                          f"(head cannot be built): {bbox_path}")
            
            if len(ill_df_single) == 0 or len(bbox_df_single) == 0:
                print(f"Warning: Skipping {ill_path} (or bbox) due to empty data after NaN drop.")
                continue

            if args.environment != 'none':
                env_df_single = pd.read_csv(env_path)
                env_df_single.dropna(inplace=True)
                all_env_sensors = env_df_single.columns.tolist()[1:]
                if not all_env_sensors:
                        raise ValueError(f"No sensor columns found in {env_path}")
                
                current_env_data = env_df_single[all_env_sensors].values.astype(np.float32).T
                self.environment_vectors_raw.append(torch.from_numpy(current_env_data))
                self.all_env_sensor_names.append(all_env_sensors)
                    
            source_dataset_id = len(self.environment_vectors_raw) - 1 if args.environment != 'none' else 0

            file_idx = test_file_idx
            test_file_idx += 1
            coord_sources.append((ill_path, ill_df_single.columns.tolist()))

            ill_df_single['timestamp_dt'] = pd.to_datetime(ill_df_single['timestamp'], format=time_format)
            bbox_df_single['timestamp_dt'] = pd.to_datetime(bbox_df_single['timestamp'], format=time_format)
            
            ill_df_single = ill_df_single.sort_values('timestamp_dt').reset_index(drop=True)
            ill_df_single['global_ill_index'] = ill_df_single.index + current_ill_index_offset

            merged_df_single = pd.merge_asof(
                bbox_df_single.sort_values('timestamp_dt'),
                ill_df_single[['timestamp_dt', 'global_ill_index']],
                on='timestamp_dt',
                direction='nearest'
            )
            merged_df_single['source_dataset_id'] = source_dataset_id
            merged_df_single['file_idx'] = file_idx

            if do_split:
                n_samples_merged = len(merged_df_single)
                num_train = int(n_samples_merged * args.train_ratio)
                num_val = int(n_samples_merged * args.val_ratio)

                if split == 'train':
                    split_df = merged_df_single.iloc[:num_train]
                elif split == 'val':
                    split_df = merged_df_single.iloc[num_train : num_train + num_val]
                else:
                    split_df = merged_df_single.iloc[num_train + num_val:]
                merged_files_to_concat.append(split_df)
            elif split == 'test' and args.adapt == 'input':
                num_adapt = int(len(merged_df_single) * args.adapt_ratio)
                merged_files_to_concat.append(merged_df_single.iloc[num_adapt:])
            else:
                merged_files_to_concat.append(merged_df_single)
            
            ill_dfs_all_concat.append(ill_df_single)
            current_ill_index_offset += len(ill_df_single)

        if not merged_files_to_concat:
             raise ValueError(f"No valid data loaded for split '{split}'. Check file paths and NaN values.")
        
        merged_df_split = pd.concat(merged_files_to_concat, ignore_index=True)

        if not ill_dfs_all_concat:
             raise ValueError("No illuminance data loaded.")
        self.illuminance_df = pd.concat(ill_dfs_all_concat, ignore_index=True)

        initial_light_rows = len(self.illuminance_df)
        self.illuminance_df.dropna(inplace=True)
        if initial_light_rows - len(self.illuminance_df) > 0:
            print(f"Dropped {initial_light_rows - len(self.illuminance_df)} rows from illuminance data due to NaN.")
            
        exclude_cols = ['timestamp', 'timestamp_dt', 'global_ill_index']
        all_sensor_columns = [col for col in self.illuminance_df.columns if col not in exclude_cols]
        
        if hasattr(args, 'selected_columns'):
            self.selected_columns = args.selected_columns
        else:
            if 0 < args.actual_num_sensors < len(all_sensor_columns):
                self.selected_columns = random.sample(all_sensor_columns, args.actual_num_sensors)
            else:
                self.selected_columns = all_sensor_columns
        
        self.mean = args.mean
        self.std = args.std

        if self.mean is not None and self.std is not None and len(all_sensor_columns) != len(self.selected_columns):
            all_columns_map = {col: i for i, col in enumerate(all_sensor_columns)}
            selected_indices = [all_columns_map[col] for col in self.selected_columns]
            if isinstance(self.mean, list):
                self.mean = [m[selected_indices] for m in self.mean]
                self.std = [s[selected_indices] for s in self.std]
            else:
                self.mean = self.mean[selected_indices]
                self.std = self.std[selected_indices]

        if self.std is None:
            self.std_eps = None
        elif isinstance(self.std, list):
            self.std_eps = [s + 1e-6 for s in self.std]
        else:
            self.std_eps = self.std + 1e-6

        if args.environment != 'none':
            self.environment_vectors = []
            all_selected_columns_map = {col: i for i, col in enumerate(self.selected_columns)}
            
            for i, (raw_vec_tensor, raw_names) in enumerate(zip(self.environment_vectors_raw, self.all_env_sensor_names)):
                raw_vec_mean = raw_vec_tensor.mean(dim=1)
                
                raw_map = {name: k for k, name in enumerate(raw_names)}
                N_selected = len(self.selected_columns)
                
                aligned_vector = torch.zeros(N_selected, dtype=torch.float32)
                
                for col_name, selected_idx in all_selected_columns_map.items():
                    if col_name in raw_map:
                        raw_idx = raw_map[col_name]
                        aligned_vector[selected_idx] = raw_vec_mean[raw_idx]
                        
                self.environment_vectors.append(aligned_vector)

        # 座標はファイルごと（マットをずらした収録は同じセンサ ID でも位置が違う）。
        # __getitem__ は target['file_idx'] でそのサンプルのファイルの座標を返す。
        has_json = {ill_path: os.path.exists(os.path.join(os.path.dirname(ill_path), SENSOR_COORDS_FILENAME))
                    for ill_path, _ in coord_sources}
        if self.selected_columns and len(set(has_json.values())) > 1:
            legacy = [p for p, v in has_json.items() if not v]
            raise ValueError(f"sensor_coords.json が無い（列名が座標の旧形式の）データが混ざっています: {legacy}。"
                             "preprocessing/preprocess_real/tools/sensor_coords.py で移行してください")
        self.sensor_coords_by_file = []
        frames = {}
        for ill_path, columns in coord_sources:
            coords, frame = load_sensor_coords(ill_path, columns, self.selected_columns)
            self.sensor_coords_by_file.append(coords)
            frames.setdefault(frame, []).append(os.path.basename(os.path.dirname(ill_path)))
        frames.pop(None, None)
        if len(frames) > 1:
            # legacy（2 m = 1）と special（[-0.6, 2.6] m）では同じ物理位置でも値が違う
            print(f"[{split}] [warn] センサ座標の正規化枠が混在しています: "
                  + "; ".join(f"{k}: {v}" for k, v in frames.items()))

        print(f"\n[{split}] Using the following {len(self.selected_columns)} sensor columns:")
        for i in range(0, len(self.selected_columns), 4):
             print("  ".join(f"{col:<20}" for col in self.selected_columns[i:i+4]))
        print("")

        self.samples = []
        # 正規化は「割る」だけでなく「シフトしてから割る」。pose.csv / bbox_label.csv は
        # メートル・原点中心なので、割るだけだと目標値の半分が負になり、sigmoid 出力の
        # モデル（models/detr.py の outputs_coord.sigmoid()）では到達できない。
        span = torch.tensor(args.space_size, dtype=torch.float32)          # X, Y, Z
        origin = torch.tensor(args.space_origin, dtype=torch.float32)      # X, Y, Z
        span_xyzxyz = span.repeat(2)
        origin_xyzxyz = origin.repeat(2)
        # target に載せるときだけ [h, w, d]（= Y, X, Z）順。消費側（models/detr.py の
        # PostProcess, util/append.py）が img_h, img_w, img_d で受けて [w, h, d] に組み替える
        # 既存の並びに合わせてある。
        orig_size_hwd = span[[1, 0, 2]].clone()
        orig_origin_hwd = origin[[1, 0, 2]].clone()
        n_clamped = 0
        n_coords = 0

        for (ill_center_idx, source_id), group in merged_df_split.groupby(['global_ill_index', 'source_dataset_id']):
            start_idx = ill_center_idx - (args.window_size - 1)
            end_idx = ill_center_idx + 1

            if start_idx < 0:
                continue

            illuminance_window = self.illuminance_df.iloc[start_idx:end_idx][self.selected_columns].values
            if illuminance_window.shape[0] != args.window_size:
                continue

            target = {}
            labels_for_frame = []
            
            if self.task == 'pose':
                keypoints_for_frame = []
                for _, row in group.iterrows():
                    kpts = [[row[f'{name}_X'], row[f'{name}_Y'], row[f'{name}_Z']] for name in self.KEYPOINT_NAMES]
                    kpts_tensor = torch.tensor(kpts, dtype=torch.float32)

                    # 合成点: neck は両肩の中点、head は取れている顔関節の平均
                    neck = (kpts_tensor[self._LEFT_SHOULDER]
                            + kpts_tensor[self._RIGHT_SHOULDER]) / 2.0
                    head_xyz = keypoints.head_point(row)
                    if head_xyz is None:
                        raise ValueError("顔関節が 1 点も取れていない行が残っています")
                    head = torch.tensor(head_xyz, dtype=torch.float32)
                    kpts_tensor = torch.cat(
                        [kpts_tensor, neck.unsqueeze(0), head.unsqueeze(0)], dim=0)

                    kpts_tensor = (kpts_tensor - origin) / span
                    n_coords += kpts_tensor.numel()
                    n_clamped += int(((kpts_tensor < 0) | (kpts_tensor > 1)).sum())
                    kpts_tensor.clamp_(0.0, 1.0)

                    keypoints_for_frame.append(kpts_tensor)
                    labels_for_frame.append(0)

                n_out = len(self.OUTPUT_KEYPOINT_NAMES)
                target['keypoints'] = torch.stack(keypoints_for_frame) if keypoints_for_frame else torch.zeros((0, n_out, 3), dtype=torch.float32)
                target['labels'] = torch.tensor(labels_for_frame, dtype=torch.long)
                
            else:
                boxes_for_frame = []
                for _, row in group.iterrows():
                    box = torch.tensor([row['x1'], row['y1'], row['z1'], row['x2'], row['y2'], row['z2']],
                                       dtype=torch.float32)
                    box = (box - origin_xyzxyz) / span_xyzxyz
                    n_coords += box.numel()
                    n_clamped += int(((box < 0) | (box > 1)).sum())
                    # cxcywhd に変換する前に clamp する。変換後に潰すと w/h/d が負になりうる。
                    box.clamp_(0.0, 1.0)
                    boxes_for_frame.append(box)
                    labels_for_frame.append(int(row['label']))

                boxes_xyzxyz = torch.stack(boxes_for_frame) if boxes_for_frame else torch.zeros((0, 6), dtype=torch.float32)
                target['boxes'] = box_xyzxyz_to_cxcywhd(boxes_xyzxyz) if len(boxes_xyzxyz) > 0 else boxes_xyzxyz
                target['labels'] = torch.tensor(labels_for_frame, dtype=torch.long)

            first_row = group.iloc[0]
            target['image_id'] = torch.tensor([ill_center_idx])
            target['orig_size'] = orig_size_hwd
            target['orig_origin'] = orig_origin_hwd
            target['image_path'] = first_row.get('image_path', first_row.get('color_filepath', ''))
            target['file_idx'] = int(first_row['file_idx'])
            if args.environment != 'none':
                target['env_vector'] = self.environment_vectors[source_id]

            self.samples.append((illuminance_window.astype(np.float32), target))

        origin_list = [f"{v:.2f}" for v in args.space_origin]
        span_list = [f"{v:.2f}" for v in args.space_size]
        print(f"[{split}] Normalization: norm = (v - origin) / span  "
              f"origin(XYZ)=[{', '.join(origin_list)}] m, span(XYZ)=[{', '.join(span_list)}] m")
        if n_clamped:
            print(f"[{split}] [warn] {n_clamped} / {n_coords} coordinates "
                  f"({100 * n_clamped / max(n_coords, 1):.3f}%) fell outside [0, 1] and were clamped. "
                  f"Widen --space_size / --space_origin if this is not negligible.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        illuminance_data, target = self.samples[idx]
        illuminance_tensor = torch.from_numpy(illuminance_data).T 

        if self.mean is not None and self.std_eps is not None:
            if isinstance(self.mean, list):
                mean_i = self.mean[target['file_idx']]
                std_eps_i = self.std_eps[target['file_idx']]
            else:
                mean_i, std_eps_i = self.mean, self.std_eps
            illuminance_tensor = (illuminance_tensor - mean_i.unsqueeze(1)) / std_eps_i.unsqueeze(1)

        return illuminance_tensor, self.sensor_coords_by_file[target['file_idx']], target

def build(image_set, args):
    illuminance_path = getattr(args, f"{image_set}_illuminance_path")
    bbox_path = getattr(args, f"{image_set}_bbox_path")
    environment_path = getattr(args, f"{image_set}_environment_path")
    
    return IlluminanceDetectionDataset(
        illuminance_csv_path=illuminance_path, 
        bbox_csv_path=bbox_path,
        environment_csv_path=environment_path,
        args=args,
        do_split=args.do_split,
        split=image_set
    )