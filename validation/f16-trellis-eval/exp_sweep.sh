#!/usr/bin/env bash
# Real-like scales: x rms 1 (post-RMSNorm hidden), suh rms 0.02. Outlier stress: 1/512 channels at 100 and 1000.
cd /tmp
python3 /opt/cuteafd-target/eval_expert_oracle.py /opt/cuteafd-target/exp_main.json \
  glm:4,3:1:1 glmf:3,4:1:1 kimik3:2,3:1:1 step5:4,6:1:1
echo "main exit $?"
python3 /opt/cuteafd-target/eval_expert_oracle.py /opt/cuteafd-target/exp_outlier.json \
  glm:4,3:1:1:100 glm:4,3:1:1:1000 kimik3:2,3:1:1:1000 step5:4,6:1:1:1000
echo "outlier exit $?"
