#!/bin/bash
# pose/DR: run_all_pose_dr2.sh の後半4つ(k8_r0/k8_r1/k8_r2/k16_r0)をGPU1で並列実行するための補助スクリプト。
# run_all_pose_dr2.sh(GPU0側)が同じ4ステップに到達する前に、GPU0側の方は手動で止めること
# (そのまま両方走らせると後半4つが二重に実行され、GPU0の計算が無駄になる)。
#
# 使い方:
#   nohup ./pose/DR/run_last4_pose_dr2_gpu1.sh > ./pose/DR/run_last4_pose_dr2_gpu1.log &
#
# 各スクリプトの標準出力はそれぞれの pose/DR/pose_dr2_k*_r*.txt に出力される
# (run_all_pose_dr2.sh が後で生成するものと同じファイル名)。

cd "$(dirname "$0")/../.."

echo "=== [1/4] pose_dr2_k8_r0_gpu1.sh (k=8, repeat=0) on GPU1 ==="
bash ./pose/DR/pose_dr2_k8_r0_gpu1.sh > ./pose/DR/pose_dr2_k8_r0.txt
echo "=== [1/4] done ==="

echo "=== [2/4] pose_dr2_k8_r1_gpu1.sh (k=8, repeat=1) on GPU1 ==="
bash ./pose/DR/pose_dr2_k8_r1_gpu1.sh > ./pose/DR/pose_dr2_k8_r1.txt
echo "=== [2/4] done ==="

echo "=== [3/4] pose_dr2_k8_r2_gpu1.sh (k=8, repeat=2) on GPU1 ==="
bash ./pose/DR/pose_dr2_k8_r2_gpu1.sh > ./pose/DR/pose_dr2_k8_r2.txt
echo "=== [3/4] done ==="

echo "=== [4/4] pose_dr2_k16_r0_gpu1.sh (k=16, repeat=0) on GPU1 ==="
bash ./pose/DR/pose_dr2_k16_r0_gpu1.sh > ./pose/DR/pose_dr2_k16_r0.txt
echo "=== [4/4] done ==="

echo "=== GPU1 last-4 pose/DR experiments finished ==="
