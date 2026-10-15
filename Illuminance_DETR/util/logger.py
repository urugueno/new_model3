from pathlib import Path
import csv

import pandas as pd

def log_results(all_final_stats: list, base_output_dir: Path, args):
    n_runs = len(all_final_stats)

    df = pd.DataFrame(all_final_stats)
    
    avg_stats = df.mean()
    std_stats = df.std()
    
    header_str = f"--- AVERAGE TEST RESULTS ({n_runs} runs) ---"
    
    avg_stats_lines = []
    std_stats_lines = []
    
    task = args.task
    if task == 'pose':
        desired_key_order = [
            'mpjpe', 'mpjdle_x', 'mpjdle_y', 'mpjdle_z', 'accuracy', 'f1_score', 
            'loss_ce', 'loss_keypoints',  'class_error',
        ]
    else:
        desired_key_order = [
            'ap', 'ap50', 'ap75', 'accuracy', 'iou', 'f1_score', 
        ]
        for c in range(args.num_classes):
            desired_key_order.extend([
                f'AP_class_{c}', f'AP50_class_{c}', f'AP75_class_{c}'
            ])
        desired_key_order.extend([
            'loss_ce', 'loss_bbox', 'class_error'
        ])
    
    sorted_keys = [key for key in desired_key_order if key in avg_stats]
    remaining_keys = [key for key in avg_stats.keys() if key not in sorted_keys]
    sorted_keys.extend(remaining_keys)

    max_key_len = max([len(key) for key in sorted_keys], default=0)
    target_len = max_key_len + 2

    for key in sorted_keys:
        avg_val = avg_stats[key]
        std_val = std_stats[key]

        formatted_key = f"'{key}'"
        avg_stats_lines.append(f"  {formatted_key:<{target_len}}: {avg_val:.4f},")
        std_stats_lines.append(f"  {formatted_key:<{target_len}}: {std_val:.4f},")

    avg_stats_str = "'Average Stats': \n" + "\n".join(avg_stats_lines) + "\n"
    std_stats_str = "'Std Deviation': \n" + "\n".join(std_stats_lines) + "\n"

    full_log_message = f"{header_str}\n{avg_stats_str}\n{std_stats_str}\n"

    print(full_log_message)

    log_txt_path = f"{base_output_dir}/log.txt"
    if Path(log_txt_path).exists():
        with open(log_txt_path, "a") as f:
            f.write(full_log_message)
            f.write("-"*80 + "\n\n\n")
    
    log_csv_path = f"{base_output_dir}/log.csv"
    if Path(log_csv_path).exists():
        try:
            avg_stats_dict = avg_stats.to_dict()
            avg_stats_dict['epoch'] = f'AVG_{n_runs}_RUNS'
            
            std_stats_dict = std_stats.to_dict()
            std_stats_dict = {f"{k}_std": v for k, v in std_stats_dict.items()}
            
            csv_row_data = {**avg_stats_dict, **std_stats_dict}

            with open(log_csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                fieldnames = next(reader)
            
            new_fieldnames = list(fieldnames)
            for key in csv_row_data.keys():
                if key not in new_fieldnames:
                    new_fieldnames.append(key)
                    
            with open(log_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction='ignore')
                
                if set(fieldnames) != set(new_fieldnames):
                    pass
                
                writer.writerow(csv_row_data)
                
        except Exception as e:
            print(f"Warning: Could not append average to log.csv. Error: {e}")