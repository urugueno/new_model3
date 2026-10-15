"""
raw/light/<データ名>/ の timestamp.csv・light_origin.csv を正規化し、
processed/light/<データ名>/ へコピーする。raw 側は一切書き換えない。

- timestamp.csv    : color/depth のパス列を raw ルート相対（<データ名>/color/xxx.jpg）に再構築し、
                     全列が同一の重複行を除く
- light_origin.csv : ヘッダーをセンサ ID（ch 名）にそろえる。古い raw の "gx,gy" 形式も ch 名に直す
                     （座標は列名に持たせず、[7.5] sensor_coords.py が sensor_coords.json に書く）
"""
import os
import sys
import csv
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import raw_dir, processed_dir
from sensor_layout import rename_to_ch


def fix_timestamp_paths(src_dir: str, dst_dir: str, folder_name: str):
    src = os.path.join(src_dir, "timestamp.csv")
    if not os.path.exists(src):
        print(f"  [skip] timestamp.csv が見つかりません: {src}")
        return

    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # パス列は毎回 raw から作り直すので、元の値の形式に依存しない
    for row in rows:
        for col, subdir in (("color_filepath", "color"), ("depth_filepath", "depth")):
            value = row.get(col)
            if value:
                row[col] = f"{folder_name}/{subdir}/{os.path.basename(value)}"

    # 収録側が同じフレームを 2 回書くことがある（全列が同一 = 同じ画像ファイル）。
    # これは落としてよい。timestamp だけが重なる行は duplicate_check で止める。
    seen, unique = set(), []
    for row in rows:
        key = tuple(row.get(c) for c in fieldnames)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    if len(unique) < len(rows):
        print(f"  [warn] 全列が同一の重複行を {len(rows) - len(unique)} 行捨てます")
        rows = unique

    dst = os.path.join(dst_dir, "timestamp.csv")
    with open(dst, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [done] timestamp.csv のパスを {folder_name}/ 相対にして書き出しました -> {dst}")

def fix_light_header(src_dir: str, dst_dir: str):
    src = os.path.join(src_dir, "light_origin.csv")
    if not os.path.exists(src):
        print(f"  [skip] light_origin.csv が見つかりません: {src}")
        return

    # 記録プロセスが閉じる前に止まると末尾が NUL 埋め＋最終行が途中切れになる。
    # NUL を除き、ヘッダーと列数が合わない行（途中切れ）は捨てる。
    with open(src, newline="") as f:
        text = f.read()
    n_nul = text.count("\x00")
    if n_nul:
        print(f"  [warn] light_origin.csv に NUL バイトが {n_nul} 個あります（除去します）")
        text = text.replace("\x00", "")
    rows = list(csv.reader(text.splitlines()))

    if not rows:
        print("  [skip] light_origin.csv が空です")
        return

    n_cols = len(rows[0])
    broken = [i for i, r in enumerate(rows[1:], start=2) if len(r) != n_cols]
    if broken:
        print(f"  [warn] 列数が {n_cols} でない行を {len(broken)} 行捨てます（行番号: {broken[:5]}）")
        rows = [rows[0]] + [r for r in rows[1:] if len(r) == n_cols]

    original_header = rows[0]

    # 列名はセンサ ID（ch 名）で固定する。座標を列名に入れるとマットをずらした収録で
    # 列名が変わってしまうため（sensor_layout.py 参照）。
    try:
        rows[0], n_renamed = rename_to_ch(rows[0], "  [error]")
    except ValueError as e:
        print(e)
        sys.exit(1)
    if n_renamed:
        print(f"  [info] グリッド番号形式の列名 {n_renamed} 個を ch 名に直しました")

    dst = os.path.join(dst_dir, "light_origin.csv")
    with open(dst, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)
    print(f"  [done] light_origin.csv を書き出しました -> {dst}")

def run(data_name: str):
    src_dir = raw_dir(data_name)
    if not os.path.isdir(src_dir):
        print(f"[error] raw フォルダが見つかりません: {src_dir}")
        sys.exit(1)
    dst_dir = processed_dir(data_name)

    print(f"[setup] {data_name}")
    fix_timestamp_paths(src_dir, dst_dir, data_name)
    fix_light_header(src_dir, dst_dir)
    print("[setup] 完了")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    run(args.data)
