#!/usr/bin/env bash
set -u
S=6144:7168:2,7168:3072:1,4096:12288:1,8192:8192:1,4096:154880:1
B=/workspace/cuteafd/third_party/sparkinfer
run() { python3 $B/benchmarks/kimi_k6_dense.py --bits 6 --codebook mcg --input-dtype $1 --shapes $S \
  --rows 1,8,16,64,2048 --samples ${3:-15} --baseline-native $B/b12x/gemm/trellis_linear/_native.py \
  --output-dir /opt/cuteafd-target/$2 ${4:+--thresholds $4}; echo "$2 exit $?"; }
python3 /opt/cuteafd-target/eval_k6_oracle.py /opt/cuteafd-target/k6-bf16 /opt/cuteafd-target/k6-f16 /opt/cuteafd-target/k6_oracle.json ${S//,/ }
echo "oracle exit $?"
run bf16 k6-bf16 50
run f16 k6-f16 50
run f16 k6-f16-d8 50 decode=8
