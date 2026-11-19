# Changes

- DeepSeek V4 Flash on SM121: accelerate the ordered native MXFP4 m4096 expert wave for the N=256, 288 and 320 Spark shards using a single-CTA route planner and expert-stationary 128-route FC1/FC2 tiles. Preserve the native weight and scratch ABI and the production FP32 fold. The measured M=2048/2752/4096 wave cuts time by 15.14% across the three widths, with exact sealed outputs. Expected changed groups: `dsv4f-spark_tp8`, `dsv4f-spark_w288`, `dsv4f-spark_w320`; no coordinator export changes. See tpurtell/cuteafd#2512.
