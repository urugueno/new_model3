#!/usr/bin/env python3
"""pose/DR: DR/dr2_comp (bbox) の体積統制フォローアップを pose タスク向けに移植したもの。
k=2/4/8/16のパターン組み合わせ(dr_subsets.json、DR/dr_comp からそのまま再利用)と
total_trials=16による体積統制は dr2_comp と同一。DR/dr2_comp との差分は以下の2点:
  - モデル: --task bbox -> --task pose (--keypoint_loss_coef / --no_iou_aware /
    --set_cost_class / --set_cost_keypoint / --class_loss_coef を付与。値は
    pose/basic_comp, pose/real_copy と同じ「DR相当」の 1/1/1)。
  - データ構造: sim_DR_ver2/DR_xxx/DR_xxx_i (NAS, 生データ) -> sim_DR_ver2_processed/
    DR_xxx/DR_xxx_i (pose.csv 等を含む前処理済みデータ)。

Usage:
    python pose_dr2_pipeline.py sync-metadata --nas-root /nas/data_2/ueno1212/light/data/sim_DR_ver2_processed \
        --out pattern_metadata_ver2.json

    python pose_dr2_pipeline.py generate --metadata pattern_metadata_ver2.json \
        --subsets dr_subsets.json --total-trials 16
"""
import argparse
import json
from pathlib import Path

TOTAL_TRIALS_DEFAULT = 16

REAL_TRAIN = "4_action_new_af/4_action_new_af_train"
REAL_VAL = "4_action_new_af/4_action_new_af_val"
REAL_TEST = "4_action_new_af/4_action_new_af_test"


def cmd_sync_metadata(args):
    nas_root = Path(args.nas_root)
    raw_meta_root = Path(args.raw_meta_root)
    data_root_prefix = args.data_root_prefix

    patterns = []
    for d in sorted(p for p in nas_root.iterdir() if p.is_dir()):
        meta_path = raw_meta_root / d.name / "pattern_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        trial_dirs = sorted(p for p in d.iterdir() if p.is_dir() and p.name.startswith(f"{d.name}_"))
        patterns.append({
            "pattern_id": int(meta["pattern_id"].split("_")[1]),
            "name": d.name,
            "family": meta["family"],
            "count": meta["count"],
            "dir": f"{data_root_prefix}/{d.name}",
            "n_trials": len(trial_dirs),
        })
    patterns.sort(key=lambda p: p["pattern_id"])

    out = {
        "_fill_in": f"{nas_root} を走査して family/count/dir/n_trials(trialサブフォルダ数)を自動反映。"
                    f"family/count は {raw_meta_root} 側の pattern_meta.json から取得。",
        "patterns": patterns,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    for p in patterns:
        print(f"pattern_id={p['pattern_id']:>2} name={p['name']} count={p['count']:>2} n_trials={p['n_trials']}")
    print(f"wrote {args.out}")


TRAIN_SCRIPT_TEMPLATE = """CUDA_VISIBLE_DEVICES={gpu} python -u Illuminance_DETR/main.py \\
    --train {real_train}  {sim_dirs} \\
    --val   {real_val} \\
    --test  {real_test} \\
    --exp_name {exp_name} \\
    --task pose --num_classes {num_classes} --keypoint_loss_coef 1 \\
    --batch_size 256 --epochs 1000 --early_stopping_patience 50 --n_runs 3 \\
    --model_mode time_sensor --scale full --environment {environment} \\
    --num_sensors 36 --k_neighbors 36 \\
    --no_iou_aware --set_cost_class 1 --set_cost_keypoint 1 --class_loss_coef 1
"""

RUN_ALL_HEADER = """#!/bin/bash
# pose/DR: 体積統制フォローアップ(sim総trial数をどのkでもT={total_trials}に揃えた比較)。
# DR/dr2_comp と同じk=2/4/8/16のパターン組み合わせを使い回し、1パターンあたりのtrial数だけを
# k=2:{t2} / k=4:{t4} / k=8:{t8} / k=16:{t16} に変えている。
# DR/dr2_comp との差分はモデル(--task pose)とデータ構造(sim_DR_ver2_processed)の2点のみ。
# pose_dr2_pipeline.py generate で本ファイルと各 pose_dr2_k*_r*.sh は再生成される。
#
# 使い方:
#   nohup ./pose/DR/run_all_pose_dr2.sh > ./pose/DR/run_all_pose_dr2.log &
#
# 各スクリプトの標準出力はそれぞれの pose/DR/pose_dr2_k*_r*.txt に出力される。

cd "$(dirname "$0")/../.."

"""


def run_order(subsets):
    order = []
    for k in ("k2", "k4", "k8", "k16"):
        for entry in sorted(subsets[k], key=lambda e: e["repeat_index"]):
            order.append((k[1:], entry["repeat_index"]))
    return order


def cmd_generate(args):
    metadata = {p["pattern_id"]: p for p in json.loads(Path(args.metadata).read_text())["patterns"]}
    subsets = json.loads(Path(args.subsets).read_text())
    total_trials = args.total_trials

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    order = run_order(subsets)
    total = len(order)
    trials_per_k = {}
    run_all_lines = []
    for step, (k, repeat_index) in enumerate(order, start=1):
        k_int = int(k)
        if total_trials % k_int != 0:
            raise ValueError(f"total_trials={total_trials} is not divisible by k={k_int}")
        trials_per_pattern = total_trials // k_int
        trials_per_k[k_int] = trials_per_pattern

        entry = next(e for e in subsets[f"k{k}"] if e["repeat_index"] == repeat_index)
        sim_dirs = []
        for pid in entry["pattern_ids"]:
            p = metadata[pid]
            if p["n_trials"] < trials_per_pattern:
                raise ValueError(
                    f"pattern_id={pid} ({p['name']}) has only {p['n_trials']} trial(s), "
                    f"but k={k_int} needs {trials_per_pattern} per pattern (total_trials={total_trials})"
                )
            for i in range(1, trials_per_pattern + 1):
                sim_dirs.append(f"{p['dir']}/{p['name']}_{i}")
        sim_dirs_str = " ".join(sim_dirs)

        exp_name = f"pose_dr2_k{k}_r{repeat_index}"
        script_name = f"{exp_name}.sh"
        txt_name = f"{exp_name}.txt"

        script_body = TRAIN_SCRIPT_TEMPLATE.format(
            gpu=args.gpu,
            real_train=args.real_train,
            sim_dirs=sim_dirs_str,
            real_val=args.real_val,
            real_test=args.real_test,
            exp_name=exp_name,
            num_classes=args.num_classes,
            environment=args.environment,
        )
        (out_dir / script_name).write_text(script_body)
        (out_dir / script_name).chmod(0o755)

        run_all_lines.append(
            f'echo "=== [{step}/{total}] {script_name} (k={k}, repeat={repeat_index}, '
            f'{trials_per_pattern} trial/pattern x {k_int} pattern = {total_trials} trials) ==="\n'
        )
        run_all_lines.append(f"bash ./pose/DR/{script_name} > ./pose/DR/{txt_name}\n")
        run_all_lines.append(f'echo "=== [{step}/{total}] done ==="\n\n')

    header = RUN_ALL_HEADER.format(
        total_trials=total_trials,
        t2=trials_per_k.get(2, "-"),
        t4=trials_per_k.get(4, "-"),
        t8=trials_per_k.get(8, "-"),
        t16=trials_per_k.get(16, "-"),
    )
    run_all_lines = [header] + run_all_lines + ['echo "=== All pose/DR (volume-controlled) experiments finished ==="\n']
    run_all_path = out_dir / "run_all_pose_dr2.sh"
    run_all_path.write_text("".join(run_all_lines))
    run_all_path.chmod(0o755)

    print(f"wrote {total} training script(s) and {run_all_path} under {out_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync-metadata", help="scan sim_DR_ver2_processed on NAS and write pattern_metadata_ver2.json")
    p_sync.add_argument("--nas-root", default="/nas/data_2/ueno1212/light/data/sim_DR_ver2_processed")
    p_sync.add_argument("--raw-meta-root", default="/nas/data_2/ueno1212/light/data/sim_DR_ver2",
                         help="family/count は processed 側に pattern_meta.json が無いため、生データ側から取得する")
    p_sync.add_argument("--data-root-prefix", default="sim_DR_ver2_processed")
    p_sync.add_argument("--out", default="pattern_metadata_ver2.json")
    p_sync.set_defaults(func=cmd_sync_metadata)

    p_gen = sub.add_parser("generate", help="write pose_dr2_k*_r*.sh + run_all_pose_dr2.sh")
    p_gen.add_argument("--metadata", default="pattern_metadata_ver2.json")
    p_gen.add_argument("--subsets", default="dr_subsets.json")
    p_gen.add_argument("--out-dir", default=".")
    p_gen.add_argument("--total-trials", type=int, default=TOTAL_TRIALS_DEFAULT)
    p_gen.add_argument("--gpu", default="0")
    p_gen.add_argument("--real-train", default=REAL_TRAIN)
    p_gen.add_argument("--real-val", default=REAL_VAL)
    p_gen.add_argument("--real-test", default=REAL_TEST)
    p_gen.add_argument("--num-classes", type=int, default=4)
    p_gen.add_argument("--environment", default="none", choices=["none", "EE", "EQ", "PE_query"])
    p_gen.set_defaults(func=cmd_generate)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
