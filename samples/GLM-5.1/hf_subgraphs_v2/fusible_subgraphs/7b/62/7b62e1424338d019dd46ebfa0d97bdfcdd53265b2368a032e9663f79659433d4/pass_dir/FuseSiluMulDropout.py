import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return (tmp_2,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_silu_mul_kernel(
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

    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)

    # SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    # = x / (1 + exp(-x))
    silu_result = in0 * tl.sigmoid(in0)
    out = silu_result * in1

    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_silu_mul(in_0, in_1):
    N = in_0.numel()
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    out = torch.empty_like(in_0)

    fused_silu_mul_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out,)

def replacement_func():
    return fused_silu_mul