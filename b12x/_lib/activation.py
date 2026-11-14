"""Shared scalar gated activation used by fused and pointer-ABI MoE kernels."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@cute.jit
def gated_activation_value(gate: Float32, up: Float32, *,
                           is_v41: cutlass.Constexpr = False,
                           has_limit: cutlass.Constexpr = False,
                           limit_value: cutlass.Constexpr = 0.0,
                           is_swigluoai: cutlass.Constexpr = False,
                           alpha: cutlass.Constexpr = 1.0,
                           offset: cutlass.Constexpr = 0.0,
                           is_situ: cutlass.Constexpr = False,
                           situ_beta: cutlass.Constexpr = 4.0,
                           situ_linear_beta: cutlass.Constexpr = 25.0,
                           fast_math: cutlass.Constexpr = False):
    """Existing dynamic MoE epilogue, before its caller's final rounding."""
    if cutlass.const_expr(is_v41):
        gate = gate.to(cutlass.BFloat16).to(Float32)
        up = up.to(cutlass.BFloat16).to(Float32)
    if cutlass.const_expr(has_limit):
        limit = Float32(limit_value)
        neg_limit = Float32(-limit_value)
        if gate > limit:
            gate = limit
        if up > limit:
            up = limit
        if up < neg_limit:
            up = neg_limit
    sigmoid_arg = gate
    up_term = up
    if cutlass.const_expr(is_swigluoai):
        sigmoid_arg = Float32(alpha) * gate
        up_term = up + Float32(offset)
    sigmoid = cute.arch.rcp_approx(
        Float32(1.0) + cute.math.exp(-sigmoid_arg, fastmath=fast_math)
    )
    if cutlass.const_expr(is_situ):
        beta = Float32(situ_beta)
        linear_beta = Float32(situ_linear_beta)
        situ_gate = beta * cute.math.tanh(gate / beta, fastmath=fast_math) * sigmoid
        situ_up = linear_beta * cute.math.tanh(up / linear_beta, fastmath=fast_math)
        return situ_gate * situ_up
    return gate * sigmoid * up_term


@dsl_user_op
def _f32_asm(asm: str, *args, loc=None, ip=None) -> Float32:
    """One PTX statement producing an f32 from f32 inputs ``$1..``."""
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(a).ir_value(loc=loc, ip=ip) for a in args], asm, "=f," + ",".join("f" for _ in args),
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def situ_vendor(gate: Float32, up: Float32, gamma: Float32, gate_cap: cutlass.Constexpr = 4.0,
                up_cap: cutlass.Constexpr = 25.0) -> Float32:
    """Inkling 2's SiTU exactly as the checkpoint's reference kernels evaluate it (vLLM d664cd1e
    ``ops/activation.py``: ``_sigmoid_approx`` :20-28, ``_softcap`` :48-58, ``_situ`` :78):

        sigmoid = rcp.approx.ftz(1 + ex2.approx.ftz(gate * -log2 e))      (mul.rn, add.rn)
        soft(x, cap) = cap * tanh.approx.f32(div.rn(x, cap))
        result = bf16(((soft(gate) * sigmoid) * soft(up)) * gamma)         (every product rn)

    ``gate``/``up`` are the BF16-exact GEMM outputs. Returns the BF16-rounded value as F32.
    """
    sigmoid = _f32_asm(
        "{ .reg .f32 t; mul.rn.f32 t, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 t, t; "
        "add.rn.f32 t, t, 0f3F800000; rcp.approx.ftz.f32 $0, t; }", gate)
    soft_gate = Float32(gate_cap) * _f32_asm("{ .reg .f32 d; div.rn.f32 d, $1, $2; tanh.approx.f32 $0, d; }",
                                             gate, Float32(gate_cap))
    soft_up = Float32(up_cap) * _f32_asm("{ .reg .f32 d; div.rn.f32 d, $1, $2; tanh.approx.f32 $0, d; }",
                                         up, Float32(up_cap))
    value = ((soft_gate * sigmoid) * soft_up) * gamma
    return value.to(cutlass.BFloat16).to(Float32)

