#!/bin/bash
# pose/DR: 体積統制フォローアップ(sim総trial数をどのkでもT=16に揃えた比較)。
# DR/dr2_comp と同じk=2/4/8/16のパターン組み合わせを使い回し、1パターンあたりのtrial数だけを
# k=2:8 / k=4:4 / k=8:2 / k=16:1 に変えている。
# DR/dr2_comp との差分はモデル(--task pose)とデータ構造(sim_DR_ver2_processed)の2点のみ。
# pose_dr2_pipeline.py generate で本ファイルと各 pose_dr2_k*_r*.sh は再生成される。
#
# 使い方:
#   nohup ./pose/DR/run_all_pose_dr2.sh > ./pose/DR/run_all_pose_dr2.log &
#
# 各スクリプトの標準出力はそれぞれの pose/DR/pose_dr2_k*_r*.txt に出力される。

cd "$(dirname "$0")/../.."

echo "=== [1/10] pose_dr2_k2_r0.sh (k=2, repeat=0, 8 trial/pattern x 2 pattern = 16 trials) ==="
bash ./pose/DR/pose_dr2_k2_r0.sh > ./pose/DR/pose_dr2_k2_r0.txt
echo "=== [1/10] done ==="

echo "=== [2/10] pose_dr2_k2_r1.sh (k=2, repeat=1, 8 trial/pattern x 2 pattern = 16 trials) ==="
bash ./pose/DR/pose_dr2_k2_r1.sh > ./pose/DR/pose_dr2_k2_r1.txt
echo "=== [2/10] done ==="

echo "=== [3/10] pose_dr2_k2_r2.sh (k=2, repeat=2, 8 trial/pattern x 2 pattern = 16 trials) ==="
bash ./pose/DR/pose_dr2_k2_r2.sh > ./pose/DR/pose_dr2_k2_r2.txt
echo "=== [3/10] done ==="

echo "=== [4/10] pose_dr2_k4_r0.sh (k=4, repeat=0, 4 trial/pattern x 4 pattern = 16 trials) ==="
bash ./pose/DR/pose_dr2_k4_r0.sh > ./pose/DR/pose_dr2_k4_r0.txt
echo "=== [4/10] done ==="

echo "=== [5/10] pose_dr2_k4_r1.sh (k=4, repeat=1, 4 trial/pattern x 4 pattern = 16 trials) ==="
bash ./pose/DR/pose_dr2_k4_r1.sh > ./pose/DR/pose_dr2_k4_r1.txt
echo "=== [5/10] done ==="

echo "=== [6/10] pose_dr2_k4_r2.sh (k=4, repeat=2, 4 trial/pattern x 4 pattern = 16 trials) ==="
bash ./pose/DR/pose_dr2_k4_r2.sh > ./pose/DR/pose_dr2_k4_r2.txt
echo "=== [6/10] done ==="

echo "=== [7/10] pose_dr2_k8_r0.sh (k=8, repeat=0, 2 trial/pattern x 8 pattern = 16 trials) ==="
bash ./pose/DR/pose_dr2_k8_r0.sh > ./pose/DR/pose_dr2_k8_r0.txt
echo "=== [7/10] done ==="

echo "=== [8/10] pose_dr2_k8_r1.sh (k=8, repeat=1, 2 trial/pattern x 8 pattern = 16 trials) ==="
bash ./pose/DR/pose_dr2_k8_r1.sh > ./pose/DR/pose_dr2_k8_r1.txt
echo "=== [8/10] done ==="

echo "=== [9/10] pose_dr2_k8_r2.sh (k=8, repeat=2, 2 trial/pattern x 8 pattern = 16 trials) ==="
bash ./pose/DR/pose_dr2_k8_r2.sh > ./pose/DR/pose_dr2_k8_r2.txt
echo "=== [9/10] done ==="

echo "=== [10/10] pose_dr2_k16_r0.sh (k=16, repeat=0, 1 trial/pattern x 16 pattern = 16 trials) ==="
bash ./pose/DR/pose_dr2_k16_r0.sh > ./pose/DR/pose_dr2_k16_r0.txt
echo "=== [10/10] done ==="

echo "=== All pose/DR (volume-controlled) experiments finished ==="
