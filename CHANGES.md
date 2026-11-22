# Changes

- SM120 EXL3 routed decode (Neon R2 of the routed MUL1 duel): a persistent
  K4 mixed-Trellis decode (`W4A16MixedTrellisSM120K4Decode`) for SM120
  coordinator buckets up to capacity 80 (K4 tier0 with K5/K6 tier1, bf16
  rotation input, route block 8), replacing the three cooperative grid
  barriers with per-block readiness and a 4-stage K ring, plus a PDL
  `W4A16TopKSumKernel` path on SM120 full-rotation sums and an exact MCG
  dequant fast path. Manifest recipes, ABIs and outputs are unchanged. On
  RTX PRO 6000 (gpu1) sealed paired replay against a stock self-pair, geomean
  stock-normalized time over rows 1-8 is 0.849 (repeat session 0.850) across
  Hy4-flash TP1/TP2 MUL1 and GLM 5.3 TP2 MCG+MUL1; floor ratio Hy4-flash TP2
  1.854 -> 1.576, GLM TP2 MCG 1.485 -> 1.210. The Qwen4 TP1 guard moved
  0.73-0.75. Every cell is bit-exact, including poisoned prefixes and
  eager == graph; zero suspected regressions; the 256-4096 prefill guard is
  within 0.9%. Expected changed placement groups: SM120 EXL3 coordinator
  routed-expert programs (K4/K5 and K4/K6 decode buckets, all SM120 EXL3
  top-k sums). See tpurtell/cuteafd#2540.

- EXL3 routed experts on SM120 and SM121: add opt-in `mul1` (`0x83DCD12D`)
  trellis codebook dequantization (`bfe`/`shf` + `mul.lo.u32` + `dp4a.u32.u32` +
  `fma.rn.f16x2`), BTX synthesis/compatibility, preparation, and paired replay
  verification across `W4A16FusedMoeKernel`, `mixed_trellis`, `mixed_trellis_k4`,
  `mixed_trellis4`, `mixed_trellis_ws`, `mixed_trellis_ws_a8`, and
  `mixed_trellis_kimi` (`K2..K6`, compact uneven-world shards, `partial_bf16`,
  and `m8192` segmented row-address views). Preserve default `mcg`
  (`0xCBAC1FED`) compiled program/cubin and LUT payload bytes, packed weight
  layouts, scratch ABIs, and runtime numeric output digests (package/manifest
  `sparkinfer_revision` provenance remains unfalsified). Warmed interleaved
  runtime-export replay measures `0.9504x` decode (`156` cells) and `0.9647x`
  prefill (`108` cells) geomean `MUL1/MCG` on SM120 (`264/264` cells `<= MCG`),
  and `0.9904x` decode (`169` cells), `0.9983x` prefill (`117` cells), and
  `0.9894x` `m8192` row-address (`8` cells) geomean `MUL1/MCG` on SM121, with
  zero own-codebook bit differences and exact prefix and eager/graph invariance.
  Expected changed default groups: none (opt-in `exl3-*-mul1` groups only). See
  tpurtell/cuteafd#2540.

- SM120 dense trellis decode: prefetch one packed K128 block per warp in
  folded K6, preserving the ordered MMA and split reduction. On RTX PRO
  6000, the stock-normalized paired chain improved 1.88% across K6/K8
  and both codebooks at M1-8 and M16. K6 MUL1 improved 17.7% for GLM Flash
  TP2 down and 18.3% for Step-5 TP2 O in that band. All 264 decode and
  280 prefill cases were bit-exact, including poisoned prefixes. Prefill
  kernel instructions and launch resources were unchanged. Expected
  changed placement group: `programs` (SM120 dense trellis).
  See tpurtell/cuteafd#2540.

- SM120 native trellis body: use specialized 128x96 and x64 suffix waves
  when they fit the runtime CTA budget, retaining the normal 128x128
  fallback. The sealed Neon R3 entry cuts paired projection-chain time
  1.78% geomean in the original 140-case duel with zero new cliff flags.
  Both timing misses pass the three-pair repeats: +0.9290% K6 GLM Flash TP2
  dense-down and +0.3475% K8 Step-5 TP2 O at 2048 rows. The stacked rerun
  is byte-exact for all 140 cases. Its diagnostic K6 GLM O adjacent flag
  comes from a larger speedup at 2048, not slower absolute 2240 latency
  in that run. The isolated repeat confirms no 2240-row regression
  (-0.57% chain, -0.79% body paired medians); 2048 chain is +0.10% with
  substantial pair-to-pair variation. Expected changed placement group:
  `programs` (SM120 dense trellis only). See tpurtell/cuteafd#2528.

- SM120 native trellis body: use 128x128 prefill for selected BF16-ingress
  K6/K8 MCG projections. Preserve packed operands, H128 stages, ordered
  K128/K1024 accumulation, decode routing and the launch ABI. All 140
  route and poisoned-prefix cases passed bitwise. At 2752 rows, median
  body time dropped 10.1%/10.8% and local projection-chain time dropped
  9.8%/10.0% (K6/K8). No server speedup is asserted. Expected changed
  placement group: `programs` (SM120 dense trellis; Kimi, GLM, GLM Flash
  and Step-5). All 14 legacy K8 128x128 head exports, including the Kimi
  head and Inkling compact-head shards, retain identical SM120 instruction
  bytes and ABI headers. See tpurtell/cuteafd#2528 for the full curve.

| Shape | K6 body at 2752 (us/row) | K8 body at 2752 (us/row) | K6 2048 to 2240 body jump | K8 2048 to 2240 body jump |
| --- | ---: | ---: | ---: | ---: |
| `glmf_o` | 0.7092 -> 0.6370 | 0.6668 -> 0.6006 | +42.16% -> +57.75% | +35.83% -> +47.28% |
| `glmf_o_tp2` | 0.3103 -> 0.2791 | 0.3118 -> 0.2806 | +13.91% -> +27.58% | +13.38% -> +27.91% |
| `glmf_dense_gate_up` | 0.4815 -> 0.4435 | 0.4807 -> 0.4398 | +27.11% -> +25.15% | +29.03% -> +26.90% |
| `glmf_dense_gate_up_tp2` | 0.2374 -> 0.2113 | 0.2396 -> 0.2113 | +0.43% -> -6.64% | +0.62% -> -6.18% |
| `glmf_dense_down` | 0.4971 -> 0.4458 | 0.5083 -> 0.4547 | +35.49% -> +48.15% | +37.02% -> +46.61% |
| `glmf_dense_down_tp2` | 0.2255 -> 0.2032 | 0.2359 -> 0.2099 | +4.90% -> +18.33% | +6.59% -> +19.60% |
| `glm_o` | 1.0154 -> 0.9127 | 1.0634 -> 0.9630 | +13.50% -> +6.03% | +12.41% -> +6.03% |
| `glm_o_tp2` | 0.4979 -> 0.4480 | 0.5425 -> 0.4822 | +26.88% -> +15.96% | +33.33% -> +22.89% |
| `mla_o_owned` | 0.8986 -> 0.8074 | 0.9034 -> 0.8041 | +30.07% -> +34.87% | +13.28% -> +17.53% |
| `step5_q` | 0.5016 -> 0.4592 | 0.5269 -> 0.4793 | +30.85% -> +28.47% | +36.90% -> +33.22% |
| `step5_q_tp2` | 0.2374 -> 0.2151 | 0.2560 -> 0.2277 | +1.80% -> -7.14% | +8.26% -> 0.00% |
| `step5_o` | 0.5038 -> 0.4487 | 0.5187 -> 0.4629 | +32.78% -> +45.78% | +37.26% -> +47.92% |
| `step5_o_tp2` | 0.2322 -> 0.2076 | 0.2493 -> 0.2218 | +9.33% -> +22.05% | +8.95% -> +20.33% |
| `muse_o_tp2` | 0.3051 -> 0.3044 | 0.3297 -> 0.3297 | +22.41% -> +22.10% | +29.55% -> +28.29% |

The 2048-to-2240 step reflects the pre-existing 128-row M tiling: 16 full
M tiles become 18, the last only half full; that is 12.5% more tile
work for 9.375% more rows before discrete CTA waves, partial-row masks,
and cache locality. Both paths have this step; the wider N tile does not
remove it and a larger speedup at 2048 can make the measured jump bigger.
The forward cliffs per width above 10% after the change are `glmf_o`,
`glmf_o_tp2`, `glmf_dense_gate_up`, `glmf_dense_down`,
`glmf_dense_down_tp2`, `glm_o_tp2`, `mla_o_owned`, `step5_q`, `step5_o`,
and `step5_o_tp2` (10 per width). `glm_o` falls below +10%, while the
already-positive steps for TP2 dense-down and TP2 O cross +10% as 2048
gets faster. The only forward 2240-to-2688
body flags in changed shapes are `glmf_dense_gate_up_tp2` and `step5_q_tp2`
(K6/K8); K6 `glmf_dense_down` also has a reverse 2688-to-2240 flag.
Counting either adjacent direction across the curve, 12/13 changed shapes
flag in each candidate width (paired original: K6 12/13, K8 13/13).
No broad row-curve pass is claimed. Original FP8 also steps on GLM Flash
full O (0.2911 -> 0.3612 us/row for the K6 session). The three slightly
slower K6 2240-row body cells are under 1% individually. `muse_o_tp2`
is an unchanged control.

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

- Spark EXL3 K4 on SM121: use one persistent work-stealing kernel for narrow
  MoE waves and noncooperative programmatic dependent launch for larger decode
  waves. Limit dispatch to seven qualified shard geometries. Warmed m1-m8
  replay on GB10 cuts latency 20.43% pooled versus rebuilt rc1 exports
  (Inkling2 TP8 23.02%, Step5 TP5 16.99%, GLM TP8 21.16%). All 144 production
  output digests and all eight prefix and eager/graph checks pass. The
  persistent path is 3.07% faster than the round-two PDL-only variant in a
  direct ABBA comparison. See tpurtell/cuteafd#2511.
