import os

# データルートの探索順は record/paths.py と同一にしてある。収録マシンには /path/to/nas が
# 無いので、最後にリポジトリ内の data/ を見る（従来の "." フォールバックより安全）。
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
DATA_ROOTS = ["/path/to/nas/data", os.path.join(_REPO, "data")]
DATA_ROOT = os.environ.get("LIGHT_DATA_ROOT") or \
    next((r for r in DATA_ROOTS if os.path.isdir(r)), ".")
RAW_ROOT = os.path.join(DATA_ROOT, "raw", "light")
PROCESSED_ROOT = os.path.join(DATA_ROOT, "processed", "light")
WEIGHT_ROOTS = ["/path/to/nas/weights"]
WEIGHT_ROOT = next((r for r in WEIGHT_ROOTS if os.path.isdir(r)), ".")
YOLO_WEIGHTS = os.path.join(WEIGHT_ROOT, "yolov8x-pose.pt")
SAM3_WEIGHTS = os.path.join(WEIGHT_ROOT, "sam3.pt")

def raw_dir(name: str) -> str:
    return os.path.join(RAW_ROOT, name)

def processed_dir(name: str) -> str:
    path = os.path.join(PROCESSED_ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path
