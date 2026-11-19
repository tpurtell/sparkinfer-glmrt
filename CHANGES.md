# Changes

- DeepSeek V4 Flash on SM121: accelerate the ordered native MXFP4 m4096
  expert wave for the N=256, 288 and 320 Spark shards with a single-CTA route
  planner and expert-stationary 128-route FC1/FC2 tiles. Preserve the packed
  weight ABI, scratch ABI and ordered FP32 fold. M=2048/2752/4096 takes 15.14%
  less time across all three widths with exact sealed outputs. Expected
  changed groups: `dsv4f-spark_tp8`, `dsv4f-spark_w288`, `dsv4f-spark_w320`;
  no coordinator export changes. See tpurtell/cuteafd#2512.

- Inkling 2 Small native NVFP4 Spark TP4, `K=4608, N=576`: specialize the
  capacity-1 and capacity-16 expert waves. Thin waves use register streaming;
  broader waves use TMA. Route preparation and input quantization share a
  launch. Preserve W4A4 arithmetic and the existing four-byte scale alignment.
  On GB10, m1-m8 replay moves from 1.7361x to 1.4178x the one-read floor
  (18.33% less time); m1 is 266.192 -> 203.760 us. Production, eager/graph, and
  decode-prefix results match byte for byte. See cuteafd issue #2512.
