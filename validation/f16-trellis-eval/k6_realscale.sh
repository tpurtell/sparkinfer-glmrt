#!/usr/bin/env bash
cd /tmp
O=/opt/cuteafd-target
S6="6144:7168:2 7168:3072:1 4096:154880:1"
SUH=0.02 python3 $O/eval_k6_oracle.py $O/k6-bf16 $O/k6-f16 $O/k6_oracle_suh02.json $S6; echo "k6 suh02 $?"
SUH=0.02 XRMS=0.05 python3 $O/eval_k6_oracle.py $O/k6-bf16 $O/k6-f16 $O/k6_oracle_suh02_x005.json 6144:7168:2 7168:3072:1; echo "k6 small $?"
BITS=8 MAX_ROWS=4096 python3 $O/eval_k6_oracle.py $O/k8-bf16 $O/k8-f16 $O/k8_oracle.json 6144:6144:1 8192:6144:2; echo "k8 suh1 $?"
BITS=8 MAX_ROWS=4096 SUH=0.02 python3 $O/eval_k6_oracle.py $O/k8-bf16 $O/k8-f16 $O/k8_oracle_suh02.json 6144:6144:1 8192:6144:2; echo "k8 suh02 $?"
