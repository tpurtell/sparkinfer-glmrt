"""V4.1 M16 fused expert-slice kernel for intermediate tiling qualification.

Consumes one expert's N256/K128 packed weights and prequantized MXFP8 rows.
Intermediate width is immutable model geometry: 576 for TP4 backbone experts
and 2304 for local dSpark experts. Packed expert strides derive from its
128-aligned storage width; live row/group counts remain runtime arguments.
The caller supplies 0..16 live rows in capacity-16 input/output storage. Width
is static model tiling; row count remains a runtime launch argument. Partial
FC2 output is FP32 and must be reduced across slices before BF16 conversion.
Grouped mode accepts runtime [expert, row_count, route_base, input_row_ids[16]]
metadata. The caller provides valid expert/input indices, disjoint contiguous
route spans and <=16 rows per group; long expert runs use multiple groups.
Nonpositive group row counts mark inactive launch slots. Output route indices
are grouped order. This low-level kernel is not a serving
binding or a replacement planner policy.

With atomic_tokens=True, grouped mode accumulates directly into a zero-initialized
flat FP32 [token_capacity * 5120] output. Metadata input row IDs also select output
tokens. This removes route/slice planes but changes FP32 addition order and flushes
subnormal atomic operands/results; callers must qualify that numerical contract.
The caller owns output clearing and stream ordering.

NVFP4 W4A8 operand contract (port plan, not yet implemented)
------------------------------------------------------------
This kernel currently consumes only the native FP4/K32 storage: ``s13``/``s2``
carry E8M0 K/32 grids, ``stage_repacked_sfb_*`` stages them, and every MMA uses
``mxfp8_mma_m16n8k32_f32_e2m1`` with operands from ``e2m1x8_to_qmma_e2m1x8``,
the residual-free six-instruction nibble spread.

ModelOpt NVFP4 keeps the same packed E2M1 payload but replaces the scale grid:
per-K/16 E4M3 block scales, decomposed for the hardware block-scale MMA into a
shared UE8M0 K/32 exponent per adjacent pair plus per-K/16 E4M3 residual
multipliers applied in-register during nibble expansion. Verified prerequisites:

* the native packer emits both grids at the extents this layout needs (see
  ``native/cuda/kernels/v41_expert_pack.cu``; K/16 doubles only the scale planes);
* the NVFP4 ``silu_v41`` weight plan resolves on SM120 and on real SM121 GB10 for
  576/top-6/384, 1152/top-6/384 and 2304/top-3/128.

Exact change set, by site in this file:

1. Constructor: ``nvfp4`` flag added; the False path must stay byte-identical.
2. Signatures: thread two residual tensors, ``w13_residual`` and
   ``down_residual`` (uint8, same logical shape as ``s13``/``s2`` but with K/16
   columns), through ``V41FusedSliceKernel.__call__``/``kernel`` and
   ``V41SlicePipeline.__call__``.
3. SMEM: the existing ``sf`` slot cannot hold the residual grids. They are twice
   as wide per row, so allocate a separate slot sized from the K/16 extent.
4. Staging needs a new helper, not a divisor argument.
   ``stage_repacked_sfb_slice`` moves four packed scale words per row with one
   ``cp_async4`` (16 bytes), which is exactly the four K/32 columns a 128-channel
   tile holds. A K/16 tile needs eight columns, i.e. two 16-byte transfers per
   row at +0 and +16, so port a ``stage_repacked_sfb_k16_slice`` that issues two
   ``cp_async4`` calls and derives its word base from the K/16 stride
   (``kernel_intermediate * 5120 // 16`` for W13, ``... // 8`` for W2, against the
   K/32 base of ``// 64`` and ``// 128``). The same applies to
   ``stage_repacked_sfb_k_slice`` for FC2.
5. FC1 MMA (the ``b[index]`` operand builds and the gate/up MMA calls): replace
   ``e2m1x8_to_qmma_e2m1x8`` with ``e2m1x8_mul_residual_to_e4m3x8`` and switch
   ``mxfp8_mma_m16n8k32_f32_e2m1`` to ``mxfp8_mma_m16n8k32_f32_e4m3``. The two
   have identical signatures, so only the call name changes. The residual operand
   is ``broadcast_f32_to_half2(fp8_e4m3_to_f32(byte))``; the byte is the K/16
   block of the current nibble, i.e. ``2*kb`` for the low nibble and ``2*kb + 1``
   for the high nibble of each K/32 pair, read at the same in-atom offset the
   scale read uses (``nf*32 + warp*8 + g``, with the up half offset by
   ``self.width`` words).
6. FC2 MMA: the same two substitutions at the FC2 operand build and its MMA,
   using the down residual at ``(kb // 4) * 128 + nf*32 + warp*8 + g`` for the
   shared K/32 exponent and the matching K/16 residual byte.
7. Export/ABI: ``python/tools/export_b12x_v41_slices_aot.py`` and
   ``export_b12x_v41_experts_aot.py`` pass the recipe through
   ``--expert-storage`` (already added for the non-slice path), and
   ``native/include/ds41rt_v41_experts.h`` gains the residual slots; the launch
   struct already reserves ``DS41RT_V41_EXPERT_POINTERS`` room.

The scale and residual reads share one index shape, so the safest first step is
to stage only the residual into the new slot and leave the scale reads untouched,
then verify against the oracle before touching FC2.

Validation must cover the RTX TP1 (2304) and TP2 (1152) geometries on SM120 and
the Spark TP4 (576, kernel width 640) geometry on SM121, against the FP32 V4.1
oracle in ``tests/moe/test_v41_nvfp4_numerics.py``. Power-of-two block scales
make the decomposition exact and isolate kernel arithmetic; real checkpoints
should be measured against the recorded decomposition envelope. A negative
control must confirm the K/32 path is unchanged after the port.

Testing strategy, in dependency order. ``tests/moe/test_v41_fused_slice.py``
already drives this kernel directly through ``cute.compile`` with a real oracle
and graph replay, so the port is testable without the AOT pipeline. Its fixture
builds K/32 grids with ``_e8m0_scale_to_w4a8_sfb_inplace``; the K/16 residual
grid needs the same 128-channel atom with twice the columns, which is the same
staging change step 4 makes, so the fixture and the kernel must agree by
construction rather than by assumption.

The cheapest decisive check does not need an NVFP4 oracle at all. Encode every
residual byte as E4M3 ``1.0``: the NVFP4 path then multiplies each nibble by one,
so feeding it the same UE8M0 K/32 grid the existing fixture already produces must
reproduce the current K/32 output *exactly*, not approximately. That isolates
staging, indexing and MMA operand form from the decomposition. Vary the residual
to ``2.0`` and the output must scale by two. Only after those two pass is it
worth building the K/16 E4M3 fixture and comparing against an NVFP4 oracle that
dequantizes ``E2M1 * E4M3(K/16) * FP32(global)``.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cuda.bindings.driver as cuda
from cutlass import Int32, Int64, Uint32, Float32
from b12x._lib.intrinsics import (
    shared_ptr_to_u32,
    get_ptr_as_int64,
    red_add_global_f32,
    e2m1x8_to_qmma_e2m1x8,
    fmax_f32,
    fmin_f32,
    mxfp8_mma_m16n8k32_f32_e2m1,
    quantize_block_fp8_mx,
    max_abs_32,
)
from b12x.moe._shared.kernels.w4a8_staging import (
    stage_repacked_b_slice,
    stage_repacked_sfb_slice,
    stage_repacked_b_k_slice,
    stage_repacked_sfb_k_slice,
)


class V41FusedSliceKernel:
    def __init__(self, width, *, grouped=False, atomic_tokens=False, intermediate=576,
                 nvfp4=False):
        assert width in (64, 128, 192)
        assert not atomic_tokens or grouped
        self.atomic_tokens = atomic_tokens
        self.grouped = grouped
        self.width = width
        # False keeps the qualified native FP4/K32 behaviour byte for byte;
        # True enables the ModelOpt NVFP4 operand contract (per-K/16 E4M3
        # residuals applied during nibble expansion, e4m3 MMA operand form).
        self.nvfp4 = bool(nvfp4)
        assert intermediate > 0 and intermediate % 32 == 0
        self.intermediate = intermediate
        self.kernel_intermediate = (intermediate + 127) // 128 * 128
        self.slices = (intermediate + width - 1) // width
        assert self.slices * width <= self.kernel_intermediate, (
            "slice exceeds packed storage"
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        xs: cute.Tensor,
        w13: cute.Tensor,
        s13: cute.Tensor,
        w2: cute.Tensor,
        s2: cute.Tensor,
        routing: cute.Tensor,
        out: cute.Tensor,
        rows: Int32,
        stream: cuda.CUstream,
        metadata: cute.Tensor | None = None,
        groups: Int32 = 1,
    ):
        self.kernel(x, xs, w13, s13, w2, s2, routing, out, rows, metadata).launch(
            grid=(self.slices, groups if self.grouped else 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        xs: cute.Tensor,
        w13: cute.Tensor,
        s13: cute.Tensor,
        w2: cute.Tensor,
        s2: cute.Tensor,
        routing: cute.Tensor,
        out: cute.Tensor,
        rows: Int32,
        metadata: cute.Tensor | None,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = tid // 32
        lane = tid % 32
        c = lane % 4
        g = lane // 4
        slice_id = cute.arch.block_idx()[0]
        expert = Int64(0)
        route_base = Int64(0)
        group = cute.arch.block_idx()[1]
        if cutlass.const_expr(self.grouped):
            expert = Int64(metadata[group, 0])
            rows = Int32(metadata[group, 1])
            route_base = Int64(metadata[group, 2])
        input_lo = g
        input_hi = g + 8
        input_scale = g + c * 8
        if cutlass.const_expr(self.grouped):
            if g < rows:
                input_lo = Int32(metadata[group, 3 + g])
            if g + 8 < rows:
                input_hi = Int32(metadata[group, 3 + g + 8])
            if c < 2 and g + c * 8 < rows:
                input_scale = Int32(metadata[group, 3 + g + c * 8])
        active = Int32(1)
        if cutlass.const_expr(self.grouped):
            active = Int32(rows > 0)
        if active > 0:
            start = slice_id * Int32(self.width)
            smem = cutlass.utils.SmemAllocator()
            # Shared weight slots also serve FC2 after the fused activation boundary.
            b = smem.allocate_tensor(
                Uint32, cute.make_layout(self.width * 32), byte_alignment=16
            )
            sf = smem.allocate_tensor(
                Uint32, cute.make_layout(max(self.width * 2, 256)), byte_alignment=16
            )
            mid = smem.allocate_tensor(
                Float32, cute.make_layout((16, self.width)), byte_alignment=16
            )
            qa = smem.allocate_tensor(
                Uint32, cute.make_layout((16, self.width // 4)), byte_alignment=16
            )
            qs = smem.allocate_tensor(
                Uint32, cute.make_layout((16, self.width // 32)), byte_alignment=16
            )
            bb = shared_ptr_to_u32(b.iterator)
            sb = shared_ptr_to_u32(sf.iterator)
            gate = cute.make_rmem_tensor((self.width // 32, 4), Float32)
            gate.fill(0)
            up = cute.make_rmem_tensor((self.width // 32, 4), Float32)
            up.fill(0)
            for kt in range(40):
                stage_repacked_b_slice(
                    w13,
                    bb,
                    expert * Int64(self.kernel_intermediate * 5120 // 4),
                    Int32(40),
                    kt,
                    start,
                    tid,
                    128,
                    self.width,
                )
                stage_repacked_b_slice(
                    w13,
                    bb + self.width * 64,
                    expert * Int64(self.kernel_intermediate * 5120 // 4),
                    Int32(40),
                    kt,
                    start + self.kernel_intermediate,
                    tid,
                    128,
                    self.width,
                )
                stage_repacked_sfb_slice(
                    s13,
                    sb,
                    expert * Int64(self.kernel_intermediate * 5120 // 64),
                    Int32(40),
                    kt,
                    start,
                    tid,
                    128,
                    self.width,
                )
                stage_repacked_sfb_slice(
                    s13,
                    sb + self.width * 4,
                    expert * Int64(self.kernel_intermediate * 5120 // 64),
                    Int32(40),
                    kt,
                    start + self.kernel_intermediate,
                    tid,
                    128,
                    self.width,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                for kb in cutlass.range_constexpr(4):
                    a0 = Uint32(0)
                    a1 = Uint32(0)
                    a2 = Uint32(0)
                    a3 = Uint32(0)
                    if g < rows:
                        a0 = x[input_lo, kt * 32 + kb * 8 + c * 2]
                        a2 = x[input_lo, kt * 32 + kb * 8 + c * 2 + 1]
                    if g + 8 < rows:
                        a1 = x[input_hi, kt * 32 + kb * 8 + c * 2]
                        a3 = x[input_hi, kt * 32 + kb * 8 + c * 2 + 1]
                    scale = Uint32(0)
                    if c < 2 and g + c * 8 < rows:
                        scale = Uint32(xs[input_scale, kt * 4 + kb])
                    for nf in cutlass.range_constexpr(self.width // 32):
                        index = ((kb * (self.width // 32) + nf) * 32 + lane) * 4 + warp
                        u0, u1 = e2m1x8_to_qmma_e2m1x8(b[index])
                        g0, g1 = e2m1x8_to_qmma_e2m1x8(b[index + self.width * 16])
                        su = sf[nf * 32 + warp * 8 + g]
                        sg = sf[self.width + nf * 32 + warp * 8 + g]
                        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                            gate[nf, 0],
                            gate[nf, 1],
                            gate[nf, 2],
                            gate[nf, 3],
                            a0,
                            a1,
                            a2,
                            a3,
                            g0,
                            g1,
                            scale,
                            sg,
                            bid_b=kb,
                        )
                        gate[nf, 0] = d0
                        gate[nf, 1] = d1
                        gate[nf, 2] = d2
                        gate[nf, 3] = d3
                        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                            up[nf, 0],
                            up[nf, 1],
                            up[nf, 2],
                            up[nf, 3],
                            a0,
                            a1,
                            a2,
                            a3,
                            u0,
                            u1,
                            scale,
                            su,
                            bid_b=kb,
                        )
                        up[nf, 0] = d0
                        up[nf, 1] = d1
                        up[nf, 2] = d2
                        up[nf, 3] = d3
                cute.arch.sync_threads()
            for nf in cutlass.range_constexpr(self.width // 32):
                for elem in cutlass.range_constexpr(4):
                    row = g + (elem // 2) * 8
                    col = nf * 32 + warp * 8 + c * 2 + elem % 2
                    gv = gate[nf, elem].to(cutlass.BFloat16).to(Float32)
                    uv = up[nf, elem].to(cutlass.BFloat16).to(Float32)
                    gv = fmin_f32(gv, Float32(10))
                    uv = fmax_f32(Float32(-10), fmin_f32(uv, Float32(10)))
                    value = Float32(0)
                    if row < rows and start + col < self.intermediate:
                        sigmoid = cute.arch.rcp_approx(
                            Float32(1) + cute.math.exp(-gv, fastmath=True)
                        )
                        value = (
                            (gv * sigmoid * uv * routing[route_base + Int64(row)])
                            .to(cutlass.BFloat16)
                            .to(Float32)
                        )
                    mid[row, col] = value
            cute.arch.sync_threads()
            for task in range(tid, 16 * (self.width // 32), 128):
                row = task // (self.width // 32)
                block = task % (self.width // 32)
                values = cute.make_rmem_tensor((32,), Float32)
                for j in cutlass.range_constexpr(32):
                    values[j] = mid[row, block * 32 + j]
                payload, scale = quantize_block_fp8_mx(
                    values, fmax_f32(max_abs_32(values), Float32(1e-4))
                )
                for j in cutlass.range_constexpr(8):
                    qa[row, block * 8 + j] = payload[j]
                qs[row, block] = scale
            cute.arch.sync_threads()
            for ot in range(40):
                stage_repacked_b_k_slice(
                    w2,
                    bb,
                    expert * Int64(self.kernel_intermediate * 5120 // 8),
                    Int32(self.kernel_intermediate // 128),
                    start,
                    ot * 128,
                    tid,
                    128,
                    self.width,
                )
                stage_repacked_sfb_k_slice(
                    s2,
                    sf,
                    expert * Int64(self.kernel_intermediate * 5120 // 128),
                    Int32(self.kernel_intermediate // 128),
                    start,
                    ot * 128,
                    tid,
                    128,
                    self.width,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                acc = cute.make_rmem_tensor((4, 4), Float32)
                acc.fill(0)
                for kb in cutlass.range_constexpr(self.width // 32):
                    a0 = qa[g, kb * 8 + c * 2]
                    a2 = qa[g, kb * 8 + c * 2 + 1]
                    a1 = qa[g + 8, kb * 8 + c * 2]
                    a3 = qa[g + 8, kb * 8 + c * 2 + 1]
                    scale = Uint32(0)
                    if c < 2:
                        scale = qs[g + c * 8, kb]
                    for nf in cutlass.range_constexpr(4):
                        index = ((kb * 4 + nf) * 32 + lane) * 4 + warp
                        b0, b1 = e2m1x8_to_qmma_e2m1x8(b[index])
                        sw = sf[(kb // 4) * 128 + nf * 32 + warp * 8 + g]
                        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                            acc[nf, 0],
                            acc[nf, 1],
                            acc[nf, 2],
                            acc[nf, 3],
                            a0,
                            a1,
                            a2,
                            a3,
                            b0,
                            b1,
                            scale,
                            sw,
                            bid_b=kb % 4,
                        )
                        acc[nf, 0] = d0
                        acc[nf, 1] = d1
                        acc[nf, 2] = d2
                        acc[nf, 3] = d3
                for nf in cutlass.range_constexpr(4):
                    for elem in cutlass.range_constexpr(4):
                        row = g + (elem // 2) * 8
                        col = ot * 128 + nf * 32 + warp * 8 + c * 2 + elem % 2
                        if cutlass.const_expr(self.grouped):
                            if row < rows:
                                if cutlass.const_expr(self.atomic_tokens):
                                    token = Int64(metadata[group, 3 + row])
                                    red_add_global_f32(
                                        get_ptr_as_int64(out, token * Int64(5120) + Int64(col)),
                                        acc[nf, elem],
                                    )
                                else:
                                    out[slice_id, route_base + Int64(row), col] = acc[nf, elem]
                        else:
                            out[slice_id, row, col] = acc[nf, elem]
                cute.arch.sync_threads()
