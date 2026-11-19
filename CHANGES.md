# Changes

- Inkling 2 Small native NVFP4 Spark TP4, `K=4608, N=576`: specialize the
  capacity-1 and capacity-16 expert waves. Thin waves use register streaming;
  broader waves use TMA. Route preparation and input quantization share a
  launch. Preserve W4A4 arithmetic and the existing four-byte scale alignment.
  On GB10, m1-m8 replay moves from 1.7361x to 1.4178x the one-read floor
  (18.33% less time); m1 is 266.192 -> 203.760 us. Production, eager/graph, and
  decode-prefix results match byte for byte. See cuteafd issue #2512.
