# Dense GEMM activation precision

Status: implemented on SM120/SM121. `gemm.blockscaled` accepts BF16 activations with
NVFP4 or MXFP8 weights and selects an activation precision for dense linear
projections. MoE routing and expert GEMMs have separate implementations.

`mode="a16"` uses the BF16 warp-MMA specialization of
`b12x/_lib/dense_gemm.py::DenseGemmKernel`. The specialization retains the dense
engine's TMA producer, shared-memory pipeline, tile scheduler, accumulator
epilogue, compiler, and launch wrapper. It loads compressed weights and their
scales into shared memory, converts weight pairs directly into BF16 MMA
registers, and accumulates in FP32. Split-K writes FP32 partials and reduces
them into BF16 output. Activations remain BF16 throughout this route.

The implementation follows the inline weight conversion and narrow-M warp-MMA
patterns in `b12x/moe/_shared/kernels/w4a16/kernel.py` and the native FP8
conversion helpers in `b12x/_lib/intrinsics.py`. The MoE engine and its prepared
scale layout are references, not dependencies of the dense launch path. Triton
is used only for supporting activation quantization and packing in
`b12x/gemm/blockscaled/_quantize.py`.

## Shared weight contract

| Recipe | Stored values | Block scales | Reconstructed weight |
| --- | --- | --- | --- |
| NVFP4 | `uint8[N,K/2]`, low nibble first | E4M3, one per 16 K values | `E2M1 * block_scale * global_scale` |
| MXFP8 | `float8_e4m3fn[N,K]` | UE8M0, one per 32 K values | `E4M3 * block_scale` |

Both activation precision routes accept the same F8_128x4-swizzled weight
scale storage. `w4a16`/`w8a16` accept its flat physical storage or native six
dimensional MMA view. `pack_weight` also supports the established compact
MXFP8 scale input, which it swizzles during weight preparation.

NVFP4 `pack_weight` borrows the packed values and scale tensors without
rewriting them. `global_scale_kind="reciprocal"` interprets the supplied weight
global scale as a quantizer multiplier and divides by it in the epilogue.
Neither mode creates a second weight-scale tensor. Global scales must be
finite and positive; reconstructed weights must fit BF16.

A16 requires contiguous, 16-byte-aligned CUDA tensors, N divisible by 8,
stored K divisible by 32, and input K divisible by 8. Its native packed BF16
conversions require PTX 9.2 (CUDA 13.3). Quantized activation execution requires
stored K divisible by 128. MXFP8's established functional path remains available
for its other supported layouts and devices.

## Preparation and graph capture

Packed and raw dense calls use the unified
[preparation lifecycle](gpu-profiles.md). `blockscaled.query_from_call` describes
the actual source/weight/output/workspace ABI; `blockscaled.plan(query,
override=...)` returns an allocation-free declaration. Prepare a request with
the real parameter tensors and a representative activation producer, then pass
the prepared plan to `blockscaled.mm(..., plan=plan)`. Standalone W4A16
likewise takes its prepared plan.

The query includes exact planned M, logical/stored K, recipe, activation mode,
scale availability and interpretation, layout/alignment eligibility, output
form, workspace form and expected-M semantics. Functional MXFP8 and provided
output/workspace calls keep their existing distinct quantization paths.
Already-quantized activations retain their supplied precision.

Explicit `activation_mode="a16"` or `"quantized"` constrains the declaration.
The per-query `BlockscaledConfig` selects the actual mode and applicable
`tile_n`, `tile_k`, and `split_k`; inactive quantized fields are `None`.
For eligible uncovered BF16 inputs, the integrated default retains A16 at M1–8
with `(tile_n, tile_k, split_k) = (128, 64, 4)`. Forced A16 without that default
route retains `(64, 64, 1)`, including M16. Native K-slice clamping and layout
restrictions still apply. Missing required activation scales are errors.

Enabled startup search races the complete eligible set on cache misses.
Explicit valid pins and completed cached choices take precedence. Disabled or
cancelled tuning still prepares the validated default; it does not publish a
measured winner. No embedded precision table or separate policy mode participates.

Timings include the actual activation production/scale, quantization and GEMM
path used by the invocation. Weight preparation remains outside timing.
Correctness, poisoned-buffer replay and quantization semantics precede timing
interpretation. Standalone GEMM-only measurements are diagnostics, not
end-to-end precision-selection evidence.

Prepare every graph-visible exact M. Scratch is caller-owned and reusable
across sequential calls; concurrent calls require disjoint output/workspace.
Capture under `session.capture()`, retain the plans for the graph lifetime,
and destroy graphs before releasing the plans or closing the session.
Capture and replay execute retained launchers without policy or compiler
resolution.
