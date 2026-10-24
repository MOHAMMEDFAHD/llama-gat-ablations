#!/usr/bin/env bash
set -euo pipefail
d="$(dirname "$0")/../data_cache/Cora/raw"; mkdir -p "$d"
for f in x tx allx y ty ally graph test.index; do
  curl -sSfL -o "$d/ind.cora.$f" "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.cora.$f"
done
echo "Cora raw files in $d"
