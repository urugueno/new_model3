#!/bin/bash
# pose/3_action 配下の実験を、前のスクリプトが完了してから次を実行する形で
# 順番に流すラッパースクリプト。
#
# 使い方:
#   nohup ./pose/3_action/run_all_pose.sh &
#
# 各スクリプトの標準出力はそれぞれの pose/3_action 配下の *.txt に出力される。

cd "$(dirname "$0")/../.."

echo "=== [1/4] 3_action/EE.sh (env=EE) ==="
bash ./pose/3_action/EE.sh > ./pose/3_action/EE.txt
echo "=== [1/4] done ==="

echo "=== [2/4] 3_action/EQ.sh (env=EQ) ==="
bash ./pose/3_action/EQ.sh > ./pose/3_action/EQ.txt
echo "=== [2/4] done ==="

echo "=== [3/4] 3_action/none.sh (env=none) ==="
bash ./pose/3_action/none.sh > ./pose/3_action/none.txt
echo "=== [3/4] done ==="

echo "=== [4/4] 3_action/only.sh (env=none, no sim data) ==="
bash ./pose/3_action/only.sh > ./pose/3_action/only.txt
echo "=== [4/4] done ==="

echo "=== All pose (3_action) experiments finished ==="
