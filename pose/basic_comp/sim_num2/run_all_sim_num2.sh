#!/bin/bash
# pose/basic_comp/sim_num2 配下の実験を、前のスクリプトが完了してから次を実行する形で
# 順番に流すラッパースクリプト。
#
# 使い方:
#   nohup ./pose/basic_comp/sim_num2/run_all_sim_num2.sh &
#
# 各スクリプトの標準出力はそれぞれの pose/basic_comp/sim_num2 配下の *.txt に出力される。

cd "$(dirname "$0")/../../.."

echo "=== [1/4] sim_num2/sim_num2_EE.sh (env=EE, sim=100w+180w) ==="
bash ./pose/basic_comp/sim_num2/sim_num2_EE.sh > ./pose/basic_comp/sim_num2/sim_num2_EE_ver1.txt
echo "=== [1/4] done ==="

echo "=== [2/4] sim_num2/sim_num2_EQ.sh (env=EQ, sim=100w+180w) ==="
bash ./pose/basic_comp/sim_num2/sim_num2_EQ.sh > ./pose/basic_comp/sim_num2/sim_num2_EQ_ver1.txt
echo "=== [2/4] done ==="

echo "=== [3/4] sim_num2/sim_num2_none.sh (env=none, sim=100w+180w) ==="
bash ./pose/basic_comp/sim_num2/sim_num2_none.sh > ./pose/basic_comp/sim_num2/sim_num2_none_ver1.txt
echo "=== [3/4] done ==="

echo "=== [4/4] sim_num2/sim_num2_only.sh (env=none, no sim data) ==="
bash ./pose/basic_comp/sim_num2/sim_num2_only.sh > ./pose/basic_comp/sim_num2/sim_num2_only_ver1.txt
echo "=== [4/4] done ==="

echo "=== All pose (basic_comp/sim_num2) experiments finished ==="
