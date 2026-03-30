import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: view(-1,1) + broadcast multiply
# ---------------------------------------------------------------------------
def pattern(in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1


def replacement_args(in_1, in_2):
    return (in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel – reserved for large tensors (>TRITON_THRESHOLD elements).
# ---------------------------------------------------------------------------
_TRITON_THRESHOLD = 65536


@triton.jit
def broadcast_mul_flat_kernel(
    in1_ptr,
    in2_ptr,
    out_ptr,
    TOTAL,
    C:     tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    row  = offs // C
    in1  = tl.load(in1_ptr + row,  mask=mask, other=0.0)
    in2  = tl.load(in2_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, in1 * in2, mask=mask)


# ---------------------------------------------------------------------------
# bfloat16-specific path: Dynamo disabled to prevent the recompilation
# spikes that inflate mean GPU time for bf16.  Plain Python dispatch is
# sufficient and avoids Triton / JIT call overhead for small tensors.
# ---------------------------------------------------------------------------
@torch._dynamo.disable
def _bf16_mul(in_1, in_2):
    N, C, TOTAL = in_2.shape[0], in_2.shape[1], in_2.shape[0] * in_2.shape[1]
    if TOTAL > _TRITON_THRESHOLD:
        out   = torch.empty_like(in_2)
        BLOCK = 1024
        grid  = ((TOTAL + BLOCK - 1) // BLOCK,)
        broadcast_mul_flat_kernel[grid](in_1, in_2, out, TOTAL, C, BLOCK=BLOCK)
        return out
    return in_1.view(-1, 1) * in_2


# ---------------------------------------------------------------------------
# Wrapper – @torch.fx.wrap → FX leaf for pattern replacement.
#
# Strategy:
#   • bfloat16 : call _bf16_mul (Dynamo-disabled) to avoid bf16-specific
#                recompilation spikes that inflate the mean latency.
#   • float32 / float16 : let Dynamo trace and optimise the native PyTorch
#                broadcast multiply – this gives better steady-state perf.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_view_broadcast_mul(in_1, in_2):
    if in_2.dtype == torch.bfloat16:
        return _bf16_mul(in_1, in_2)
    # fp32 / fp16: native path, Dynamo can optimise
    N, C, TOTAL = in_2.shape[0], in_2.shape[1], in_2.shape[0] * in_2.shape[1]
    if TOTAL > _TRITON_THRESHOLD:
        out   = torch.empty_like(in_2)
        BLOCK = 1024
        grid  = ((TOTAL + BLOCK - 1) // BLOCK,)
        broadcast_mul_flat_kernel[grid](in_1, in_2, out, TOTAL, C, BLOCK=BLOCK)
        return out
    return in_1.view(-1, 1) * in_2


def replacement_func():
    return fused_view_broadcast_mul