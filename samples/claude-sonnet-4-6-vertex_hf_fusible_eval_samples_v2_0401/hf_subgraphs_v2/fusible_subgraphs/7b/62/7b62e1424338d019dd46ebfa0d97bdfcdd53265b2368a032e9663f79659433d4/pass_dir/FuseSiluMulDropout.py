import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: in_0 * in_1 (bare multiply)
#   Matches tmp_0 * in_1 where tmp_0 = silu(raw_in_0).
#   The silu op is NOT in the pattern - it remains in the graph.
#   This gives a Triton mul kernel that replaces the elementwise multiply.
#   The replacement receives (silu_output, in_1) and computes their product.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    return in_0 * in_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: elementwise multiply
#   Pattern matches tmp_0 * in_1 where tmp_0 = silu(raw_in_0).
#   in_0 is the silu output, in_1 is the gate tensor.
# ---------------------------------------------------------------------------
@triton.jit
def _mul_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in0_ptr + offsets, mask=mask)
    y = tl.load(in1_ptr + offsets, mask=mask)
    out = x * y
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_silu_mul(in_0, in_1):
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = triton.cdiv(n_elements, BLOCK_SIZE)
    out = torch.empty_like(in_0)

    _mul_kernel[(num_programs,)](
        in_0, in_1, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return out


def replacement_func():
    return fused_silu_mul