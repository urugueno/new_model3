import os
import sys
import pandas as pd
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import processed_dir

parser = argparse.ArgumentParser(description="Check for duplicate timestamps in timestamp.csv.")
parser.add_argument("--data", type=str, required=True)
args = parser.parse_args()

file_path = os.path.join(processed_dir(args.data), "timestamp.csv")

try:
    df = pd.read_csv(file_path)
    duplicates = df[df.duplicated(subset=["timestamp"], keep=False)]

    if len(duplicates) > 0:
        print(f"[warn] 重複が見つかりました。重複している行数: {len(duplicates)}")
        print(duplicates.head(20))
        sys.exit(1)
    else:
        print("[ok] timestamp列に重複はありません。")

except FileNotFoundError:
    print(f"[error] ファイルが見つかりません: {file_path}")
    sys.exit(1)
except KeyError:
    print(f"[error] 'timestamp' 列が見つかりません。存在する列: {df.columns.tolist()}")
    sys.exit(1)
except Exception as e:
    print(f"[error] 予期せぬエラー: {e}")
    sys.exit(1)
