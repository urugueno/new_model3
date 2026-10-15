import pandas as pd
import csv
import argparse
import sys
from datetime import datetime, timedelta
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import raw_dir, processed_dir
# センサ ID の定義は real と共有する（sensor_layout.py は config を import しないので衝突しない）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "preprocess_real"))
from sensor_layout import GRID_TO_CH, rename_to_ch

# 水平方向(X/Y)の bbox 固定サイズ（メートル）。高さ方向(Z)は関節 Z の robust max で動的に決める。
# real（preprocess_real.py の BBOX_SIZE_M）と同じ値にしておくこと。
BBOX_SIZE_M = 0.5


def load_sensor_df(input_csv_path, max_coord, data_name):
    """
    light / environment の CSV を読み、列名を real と同じ形式にそろえる。戻り値は (df, ch 形式か)。

    Blender の出力は列名がグリッド番号 "gx,gy"。real と同じ 6x6・0.4 m 間隔のマット
    （2x2 m の部屋）なら、real と同じセンサ ID（ch0_pi2 など）に置き換える。座標は main.py が
    real の tools/sensor_coords.py で sensor_coords.json に書き出す。
    gx は pose の +X、gy は +Y の向きで、real のグリッド定義（sensor_layout.py）と一致する
    （影になったセンサと腰位置の相関で確認済み: gx-X 0.95, gy-Y 0.94）。

    それ以外の配置（4x4 / 6x6 m の部屋、ランダム配置）は real のセンサ ID に対応しないので、
    従来どおり列名を max_coord で割った正規化座標 "x,y" にする。この形式は real と混ぜて学習できない。
    """
    df = pd.read_csv(input_csv_path)
    columns = df.columns.tolist()
    # 浮動小数の列名（ランダム配置のメートル座標）を rename_to_ch に渡すと、旧 real の正規化値と
    # 区別できずに誤変換しうるので、整数グリッド番号のときだけ ch 形式にする
    is_grid = all(c.replace(" ", "") in GRID_TO_CH for c in columns[1:])
    if is_grid:
        df.columns, _ = rename_to_ch(columns, f"[sensor:{data_name}]")
        return df, True
    if max_coord is None:
        sys.exit(f"[error] {os.path.basename(input_csv_path)} の列名が real の 6x6 グリッドに対応しません。"
                 f"従来形式で処理するには --max_coord を指定してください（real とは混ぜられません）")
    print(f"[warn] {os.path.basename(input_csv_path)} は real の 6x6 グリッドではないため、"
          f"列名を max_coord={max_coord} で正規化した座標にします（real と混ぜて学習できません）")
    return normalize_light_header_df(input_csv_path, max_coord), False


def normalize_light_header_df(input_csv_path, max_val):
    with open(input_csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    coordinate_columns = header[1:]
    max_coordinate = float(max_val)

    new_header = [header[0]]
    for coord in coordinate_columns:
        try:
            x, y = map(float, coord.split(','))
            normalized_x = x / max_coordinate
            normalized_y = y / max_coordinate
            new_header.append(f'{normalized_x},{normalized_y}')
        except (ValueError, IndexError):
            new_header.append(coord)
    
    df = pd.read_csv(input_csv_path)
    df.columns = new_header
    return df

def generate_pose_timestamps_and_map(df, start_date_str, fps):
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    time_increment = timedelta(seconds=1/fps)
    
    unique_original_timestamps = df.iloc[:, 0].unique()
    
    timestamp_map = {}
    for i, original_ts in enumerate(unique_original_timestamps):
        new_datetime = start_date + i * time_increment
        formatted_timestamp = new_datetime.strftime('%Y-%m-%d %H:%M:%S %f')
        timestamp_map[original_ts] = formatted_timestamp
        
    df[df.columns[0]] = df[df.columns[0]].astype(object)
    df.iloc[:, 0] = df.iloc[:, 0].map(timestamp_map)
    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
    
    return df, timestamp_map

def apply_mapped_timestamps(df, pose_timestamp_map, data_name="Data"):
    pose_map_float_keys = {float(k): v for k, v in pose_timestamp_map.items()}
    sorted_pose_keys = sorted(pose_map_float_keys.keys())
    
    new_timestamps = []
    
    for ts_orig in df.iloc[:, 0]:
        ts = float(ts_orig)
        
        if ts in pose_map_float_keys:
            new_timestamps.append(pose_map_float_keys[ts])
            continue

        key_before = None
        key_after = None
        for k in sorted_pose_keys:
            if k < ts:
                key_before = k
            else:
                key_after = k
                break
        
        if key_before is not None and key_after is not None:
            ts_str_before = pose_map_float_keys[key_before]
            ts_str_after = pose_map_float_keys[key_after]
            dt_before = datetime.strptime(ts_str_before, '%Y-%m-%d %H:%M:%S %f')
            dt_after = datetime.strptime(ts_str_after, '%Y-%m-%d %H:%M:%S %f')
            
            key_range = key_after - key_before
            interpolation_factor = (ts - key_before) / key_range
            
            time_diff = dt_after - dt_before
            interpolated_dt = dt_before + time_diff * interpolation_factor
            
            new_timestamps.append(interpolated_dt.strftime('%Y-%m-%d %H:%M:%S %f'))
        else:
            print(f"Warning: Timestamp {ts} in {data_name} is out of Pose range and could not be processed.")
            new_timestamps.append(None)

    df[df.columns[0]] = df[df.columns[0]].astype(object)
    df.iloc[:, 0] = new_timestamps
    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
    df.dropna(subset=['timestamp'], inplace=True)
    
    return df

def apply_coco_format_and_add_color_path(df):
    rename_map = {
        "Head": "nose",
        "LeftArm": "left_shoulder",
        "RightArm": "right_shoulder",
        "LeftForeArm": "left_elbow",
        "RightForeArm": "right_elbow",
        "LeftHand": "left_wrist",
        "RightHand": "right_wrist",
        "LeftUpLeg": "left_hip",
        "RightUpLeg": "right_hip",
        "LeftLeg": "left_knee",
        "RightLeg": "right_knee",
        "LeftFoot": "left_ankle",
        "RightFoot": "right_ankle",
        "Neck": "neck"
    }
    
    new_columns = []
    for col in df.columns:
        if '_' in col and col != 'timestamp':
            parts = col.rsplit('_', 1)
            if len(parts) == 2:
                joint_name, axis = parts
                if joint_name in rename_map:
                    new_columns.append(f"{rename_map[joint_name]}_{axis}")
                else:
                    new_columns.append(col)
            else:
                new_columns.append(col)
        else:
            new_columns.append(col)
            
    df.columns = new_columns
    df['color_filepath'] = 'images/' + df['timestamp'].astype(str) + '.jpg'

    # real の pose.csv と同じ列だけ残す。Blender の骨（Hips, Spine など）と neck は落とす
    # （neck は学習側 util/keypoints.py が肩から合成する）。
    joints = [j for j in rename_map.values() if j != "neck"]
    keep = ['timestamp', 'color_filepath'] + [f"{j}_{a}" for j in joints for a in "XYZ"
                                               if f"{j}_{a}" in df.columns]
    return df[keep]

def save_bbox_graph(bbox_df, graph_dir):
    """
    生成した（メートル）bbox 座標のフレーム推移を検証用に描画する:
    X (x1/x2), Y (y1/y2), Z (z1/z2)。graph/bbox_check.png に保存。
    """
    try:
        os.makedirs(graph_dir, exist_ok=True)
        df = bbox_df.sort_values('timestamp')
        idx = range(len(df))

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(idx, df['x1'], label='x1', color='blue')
        axes[0].plot(idx, df['x2'], label='x2', color='red')
        axes[0].set_ylabel('metre'); axes[0].set_title('bbox X (x1/x2)'); axes[0].legend()

        axes[1].plot(idx, df['y1'], label='y1', color='blue')
        axes[1].plot(idx, df['y2'], label='y2', color='red')
        axes[1].set_ylabel('metre'); axes[1].set_title('bbox Y (y1/y2)'); axes[1].legend()

        axes[2].plot(idx, df['z1'], label='z1', color='green')
        axes[2].plot(idx, df['z2'], label='z2', color='orange')
        axes[2].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        axes[2].set_ylabel('metre'); axes[2].set_xlabel('frame'); axes[2].set_title('bbox Z (z1/z2)'); axes[2].legend()

        plt.tight_layout()
        out_path = os.path.join(graph_dir, 'bbox_check.png')
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"[bbox] bbox 座標グラフを保存しました -> graph/bbox_check.png")
    except Exception as e:
        print(f"[bbox] Warning: failed to save bbox graph: {e}")


def get_robust_min_max(series, iqr_scale=1.5):
    """IQR で外れ値を除いた min/max。preprocess_real.py の同名関数と同じ処理。"""
    if len(series) < 4:
        return series.min(), series.max()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    filtered = series[(series >= q1 - iqr * iqr_scale) & (series <= q3 + iqr * iqr_scale)]
    if filtered.empty:
        return series.min(), series.max()
    return filtered.min(), filtered.max()


def generate_bounding_boxes_from_df(df, output_csv_path):
    """
    pose DataFrame（メートル単位・1 人・_X/_Y/_Z 列・timestamp・color_filepath）から
    フレームごとに 3D bbox を生成し bbox_label.csv を出力する。

    - 水平方向(X/Y): hip 中心に固定サイズ BBOX_SIZE_M の箱
    - 高さ方向(Z)  : z1 = 0（床）、z2 = 全関節 Z の robust max
    real（preprocess_real.py の --fix-z）と同じ定義にしてある。
    """
    print(f"[bbox] Generating bounding boxes -> {os.path.basename(output_csv_path)}")
    required_cols = ['timestamp', 'color_filepath',
                     'left_hip_X', 'right_hip_X', 'left_hip_Y', 'right_hip_Y']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[bbox] Error: required columns not found: {missing}")
        return False

    z_columns = [col for col in df.columns if col.endswith('_Z')]
    if not z_columns:
        print("[bbox] Error: no _Z columns found.")
        return False

    half = BBOX_SIZE_M / 2
    bbox_rows = []
    for _, row in df.iterrows():
        center_x = (row['left_hip_X'] + row['right_hip_X']) / 2
        center_y = (row['left_hip_Y'] + row['right_hip_Y']) / 2

        z_values = row[z_columns].astype(float)
        z1, z2 = 0.0, float(get_robust_min_max(z_values)[1])

        bbox_rows.append({
            'timestamp': row['timestamp'],
            'label': 0,
            'x1': center_x - half, 'y1': center_y - half, 'z1': z1,
            'x2': center_x + half, 'y2': center_y + half, 'z2': z2,
            'confidence': 1.0,
            'image_path': row['color_filepath'],
            'center_x': center_x,
            'center_y': center_y,
        })

    bbox_df = pd.DataFrame(bbox_rows)
    bbox_df.to_csv(output_csv_path, index=False, float_format='%.6f')
    print(f"[bbox] {len(bbox_rows)} frames written.")

    graph_dir = os.path.join(os.path.dirname(output_csv_path), 'graph')
    save_bbox_graph(bbox_df, graph_dir)
    return True

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument('--fps', type=int, required=True)
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--max_coord', type=float, default=None,
                        help='real の 6x6 グリッドでない配置のときだけ必要（列名をこの値で正規化する）')

    args = parser.parse_args()

    raw_path = raw_dir(args.data)
    data_path = processed_dir(args.data)

    pose_origin_path = os.path.join(raw_path, 'pose_origin.csv')
    light_origin_path = os.path.join(raw_path, 'light_origin.csv')
    timestamp_origin_path = os.path.join(raw_path, 'timestamp_origin.csv')
    environment_origin_path = os.path.join(raw_path, 'environment_origin.csv')

    pose_final_path = os.path.join(data_path, 'pose.csv')
    light_final_path = os.path.join(data_path, 'light.csv')
    timestamp_final_path = os.path.join(data_path, 'timestamp.csv')
    environment_final_path = os.path.join(data_path, 'environment.csv')
    bbox_final_path = os.path.join(data_path, 'bbox_label.csv')

    # Pose is kept in the source unit (metres): X/Y are floor-plane coordinates with the
    # origin at the room centre, Z is height above the floor. This is the same frame as real
    # (preprocess_real.py with --area_size 2.0): the 2x2 m sensor mat spans 0..2 m and the
    # pose spans -1..1 m.
    pose_df = pd.read_csv(pose_origin_path)
    pose_df, pose_timestamp_map = generate_pose_timestamps_and_map(pose_df, args.date, args.fps)
    pose_df = apply_coco_format_and_add_color_path(pose_df)
    pose_df.to_csv(pose_final_path, index=False, float_format='%.6f')

    generate_bounding_boxes_from_df(pose_df, bbox_final_path)

    light_df, _ = load_sensor_df(light_origin_path, args.max_coord, args.data)
    light_df = apply_mapped_timestamps(light_df, pose_timestamp_map, data_name="Light Data")
    light_df.to_csv(light_final_path, index=False, float_format='%.6f')

    if os.path.exists(environment_origin_path):
        env_df, _ = load_sensor_df(environment_origin_path, args.max_coord, args.data)
        env_df = apply_mapped_timestamps(env_df, pose_timestamp_map, data_name="Environment Data")
        env_df.to_csv(environment_final_path, index=False, float_format='%.6f')

    if os.path.exists(timestamp_origin_path):
        timestamp_df = pd.read_csv(timestamp_origin_path)
        timestamp_df = apply_mapped_timestamps(timestamp_df, pose_timestamp_map, data_name="Timestamp Data")
        timestamp_df.to_csv(timestamp_final_path, index=False)

    print("Final output files:")
    print(f"  - {pose_final_path}")
    print(f"  - {bbox_final_path}")
    print(f"  - {light_final_path}")
    if os.path.exists(environment_final_path):
        print(f"  - {environment_final_path}")
    if os.path.exists(timestamp_final_path):
        print(f"  - {timestamp_final_path}")

if __name__ == '__main__':
    main()