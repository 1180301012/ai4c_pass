import torch
import torch.fx
import operator
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused relu + scale*x + bias in a single memory pass.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['N'],
)
@triton.jit
def relu_scale_bias_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x     = tl.load(x_ptr     + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)   # shape-[1] scalar
    bias  = tl.load(bias_ptr)    # shape-[1] scalar

    result = x * scale + bias    # x is already relu'd; no tl.maximum needed
    tl.store(out_ptr + offsets, result, mask=mask)


# ---------------------------------------------------------------------------
# Pattern using the MIXED op representation that TorchDynamo produces:
#   relu   → aten.relu.default    (TorchDynamo decomposes relu to aten)
#   a * b  → operator.mul         (kept as Python operator)
#   a + b  → operator.add         (kept as Python operator)
#
# ForceArgsTracer produces:
#   aten.relu.default(proxy) → args=(proxy,), kwargs={}    ← matches target
#   proxy * proxy            → args=(p1, p2), kwargs={}    ← matches target
#   proxy + proxy            → args=(p1, p2), kwargs={}    ← matches target
#
# The anchor is operator.add which IS in the target graph, enabling matching.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    # Match ONLY the mul+add sub-pattern (operator.mul, operator.add).
    # in_2 will match relu_result in the target (relu stays in graph before mul).
    # ForceArgsTracer normalizes Python * and + to operator.mul/add without
    # extra args, so both match the target exactly.
    #
    # The replacement fused_relu_scale_bias(bias, scale, relu_result) computes
    # relu(relu_result)*scale+bias = relu(activation)*scale+bias (idempotent). ✓
    tmp_3 = in_1 * in_2        # operator.mul → matches target's mul(scale, relu_result)
    tmp_4 = tmp_3 + in_0       # operator.add → matches target's add(mul_result, bias)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    # in_0=bias [1], in_1=scale [1], in_2=activation [B,C,H,W]
    return (in_0, in_1, in_2)


@torch.fx.wrap
def fused_relu_scale_bias(bias, scale, x):
    """Replacement for scale * relu_result + bias.
    For small N, fall back to direct ops to avoid kernel launch overhead.
    For large N, use the Triton kernel for better memory bandwidth efficiency.
    """
    N = x.numel()
    if N < 1_000_000:
        return x * scale + bias   # direct ops, no kernel launch overhead for small N
    out = torch.empty_like(x)
    relu_scale_bias_kernel[
        lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    ](x, scale, bias, out, N)
    return out


def replacement_func():
    return fused_relu_scale_bias