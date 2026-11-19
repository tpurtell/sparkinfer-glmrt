# Changes

## Spark EXL3 K4 decode

Use one persistent work-stealing kernel for narrow SM121 K4 MoE waves, with
noncooperative programmatic dependent launch for larger decode waves. Limit
the new path to the seven qualified shard geometries; other layouts retain
the existing kernels.

Cassowary (GB10, SM121), warmed native graph replay, m1-m8 matched to rebuilt
rc1 exports: latency falls 20.43% pooled (Inkling2 TP8 23.02%, Step5 TP5
16.99%, GLM TP8 21.16%). All 144 production output digests and all eight
m1/m8/m80 prefix and eager/graph checks pass. A direct ABBA comparison put
the persistent path 3.07% ahead of the round-two PDL-only variant.
