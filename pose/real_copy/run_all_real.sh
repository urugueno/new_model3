#!/bin/bash
# pose/real_copy 配下の実験を、前のスクリプトが完了してから次を実行する形で
# 順番に流すラッパースクリプト。
#
# 使い方:
#   nohup ./pose/real_copy/run_all_real.sh &
#
# 各スクリプトの標準出力はそれぞれの pose/real_copy 配下の *.txt に出力される。

cd "$(dirname "$0")/../.."

echo "=== [1/3] real_copy/real_EE.sh (env=EE) ==="
bash ./pose/real_copy/real_EE.sh > ./pose/real_copy/real_EE.txt
echo "=== [1/3] done ==="

echo "=== [2/3] real_copy/real_EQ.sh (env=EQ) ==="
bash ./pose/real_copy/real_EQ.sh > ./pose/real_copy/real_EQ.txt
echo "=== [2/3] done ==="

echo "=== [3/3] real_copy/real_none.sh (env=none) ==="
bash ./pose/real_copy/real_none.sh > ./pose/real_copy/real_none.txt
echo "=== [3/3] done ==="

echo "=== All pose (real_copy) experiments finished ==="
