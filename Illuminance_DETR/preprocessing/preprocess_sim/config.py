import os

# preprocessing/preprocess_real/config.py と同じ探索順にする。sim の main.py は real の
# tools/sensor_coords.py を呼ぶので、両者が同じ PROCESSED_ROOT を指していないといけない。
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
DATA_ROOTS = ["/path/to/nas/data", os.path.join(_REPO, "data")]
DATA_ROOT = os.environ.get("LIGHT_DATA_ROOT") or \
    next((r for r in DATA_ROOTS if os.path.isdir(r)), ".")
RAW_ROOT = os.path.join(DATA_ROOT, "raw", "light")
PROCESSED_ROOT = os.path.join(DATA_ROOT, "processed", "light")
WEIGHT_ROOTS = ["/path/to/nas/weights"]
WEIGHT_ROOT = next((r for r in WEIGHT_ROOTS if os.path.isdir(r)), ".")
YOLO_WEIGHTS = os.path.join(WEIGHT_ROOT, "yolov8x-pose.pt")

def raw_dir(name: str) -> str:
    return os.path.join(RAW_ROOT, name)

def processed_dir(name: str) -> str:
    path = os.path.join(PROCESSED_ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path
