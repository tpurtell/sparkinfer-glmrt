#!/usr/bin/env bash
# compile_k6.sh DTYPE OUTDIR SHAPES [THRESHOLDS]
set -euo pipefail
dtype=$1 out=$2 shapes=$3 thr=${4:-}
cd /tmp
args=(--bits 6 --codebook mcg --input-dtype "$dtype" --shapes "$shapes" --rows 1,8,16,64,2048 --samples 50
      --baseline-native /workspace/cuteafd/third_party/sparkinfer/b12x/gemm/trellis_linear/_native.py
      --compile-only --output-dir "/opt/cuteafd-target/$out")
[[ -n "$thr" ]] && args+=(--thresholds "$thr")
exec /workspace/cuteafd/scripts/build/build-slot.sh python3 /workspace/cuteafd/third_party/sparkinfer/benchmarks/kimi_k6_dense.py "${args[@]}"
