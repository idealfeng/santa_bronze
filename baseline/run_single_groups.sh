#!/usr/bin/env bash
set -euo pipefail
in_csv="$1"
out_csv="$2"

cd /mnt/d/Paper/kaggle/santa/baseline
g++ -O3 -march=native -std=c++17 -fopenmp -o single_group_optimizer single_group_optimizer.cpp

in_real="$(readlink -f "$in_csv" || echo "$in_csv")"
sub_real="$(readlink -f submission.csv || echo submission.csv)"
if [[ "$in_real" != "$sub_real" ]]; then
  cp "$in_csv" submission.csv
fi
for n in $(seq 1 35); do
  ./single_group_optimizer -g "$n" -i submission.csv -o tmp.csv -n 120000 -r 1024
  mv tmp.csv submission.csv
done
cp submission.csv "$out_csv"
