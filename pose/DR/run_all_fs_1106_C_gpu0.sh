#!/bin/bash
# pose/DR fs_1106: sim_fs_1106_C_D_af の C系5パターンをGPU0で直列実行する。
# (D系はGPU1側の run_all_fs_1106_D_gpu1.sh)。
#
# 使い方:
#   nohup ./pose/DR/run_all_fs_1106_C_gpu0.sh > ./pose/DR/run_all_fs_1106_C_gpu0.log &
#
# 各スクリプトの標準出力はそれぞれの pose/DR/fs_1106_C_*.txt に出力される。

cd "$(dirname "$0")/../.."

echo "=== [1/5] fs_1106_C_ver1_DR_004_1.sh ==="
bash ./pose/DR/fs_1106_C_ver1_DR_004_1.sh > ./pose/DR/fs_1106_C_ver1_DR_004_1.txt
echo "=== [1/5] done ==="

echo "=== [2/5] fs_1106_C_ver2_DR_001_1.sh ==="
bash ./pose/DR/fs_1106_C_ver2_DR_001_1.sh > ./pose/DR/fs_1106_C_ver2_DR_001_1.txt
echo "=== [2/5] done ==="

echo "=== [3/5] fs_1106_C_ver3_DR_001_1.sh ==="
bash ./pose/DR/fs_1106_C_ver3_DR_001_1.sh > ./pose/DR/fs_1106_C_ver3_DR_001_1.txt
echo "=== [3/5] done ==="

echo "=== [4/5] fs_1106_C_ver4_DR_001_1.sh ==="
bash ./pose/DR/fs_1106_C_ver4_DR_001_1.sh > ./pose/DR/fs_1106_C_ver4_DR_001_1.txt
echo "=== [4/5] done ==="

echo "=== [5/5] fs_1106_C_ver5_DR_005_1.sh ==="
bash ./pose/DR/fs_1106_C_ver5_DR_005_1.sh > ./pose/DR/fs_1106_C_ver5_DR_005_1.txt
echo "=== [5/5] done ==="

echo "=== GPU0 pose/DR fs_1106 C (5/5) experiments finished ==="
