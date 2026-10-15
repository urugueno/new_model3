#!/bin/bash
# pose/basic_comp/sim_num1 配下の実験を、前のスクリプトが完了してから次を実行する形で
# 順番に流すラッパースクリプト。
#
# 使い方:
#   nohup ./pose/basic_comp/sim_num1/run_all_sim_num1.sh &
#
# 各スクリプトの標準出力はそれぞれの pose/basic_comp/sim_num1 配下の *.txt に出力される。

cd "$(dirname "$0")/../../.."

echo "=== [1/8] sim_num1/sim1_EE.sh (env=EE, sim=100w) ==="
bash ./pose/basic_comp/sim_num1/sim1_EE.sh > ./pose/basic_comp/sim_num1/sim1_EE_ver1.txt
echo "=== [1/8] done ==="

echo "=== [2/8] sim_num1/sim1_EQ.sh (env=EQ, sim=100w) ==="
bash ./pose/basic_comp/sim_num1/sim1_EQ.sh > ./pose/basic_comp/sim_num1/sim1_EQ_ver1.txt
echo "=== [2/8] done ==="

echo "=== [3/8] sim_num1/sim1_none.sh (env=none, sim=100w) ==="
bash ./pose/basic_comp/sim_num1/sim1_none.sh > ./pose/basic_comp/sim_num1/sim1_none_ver1.txt
echo "=== [3/8] done ==="

echo "=== [4/8] sim_num1/sim1_only.sh (env=none, no sim data) ==="
bash ./pose/basic_comp/sim_num1/sim1_only.sh > ./pose/basic_comp/sim_num1/sim1_only_ver1.txt
echo "=== [4/8] done ==="

echo "=== [5/8] sim_num1/sim2_EE.sh (env=EE, sim=180w) ==="
bash ./pose/basic_comp/sim_num1/sim2_EE.sh > ./pose/basic_comp/sim_num1/sim2_EE_ver1.txt
echo "=== [5/8] done ==="

echo "=== [6/8] sim_num1/sim2_EQ.sh (env=EQ, sim=180w) ==="
bash ./pose/basic_comp/sim_num1/sim2_EQ.sh > ./pose/basic_comp/sim_num1/sim2_EQ_ver1.txt
echo "=== [6/8] done ==="

echo "=== [7/8] sim_num1/sim2_none.sh (env=none, sim=180w) ==="
bash ./pose/basic_comp/sim_num1/sim2_none.sh > ./pose/basic_comp/sim_num1/sim2_none_ver1.txt
echo "=== [7/8] done ==="

echo "=== [8/8] sim_num1/sim2_only.sh (env=none, no sim data) ==="
bash ./pose/basic_comp/sim_num1/sim2_only.sh > ./pose/basic_comp/sim_num1/sim2_only_ver1.txt
echo "=== [8/8] done ==="

echo "=== All pose (basic_comp/sim_num1) experiments finished ==="
