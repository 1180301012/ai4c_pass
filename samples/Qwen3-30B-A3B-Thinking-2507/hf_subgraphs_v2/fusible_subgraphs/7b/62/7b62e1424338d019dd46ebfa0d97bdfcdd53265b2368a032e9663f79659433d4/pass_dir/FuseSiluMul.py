import torch
import triton
import triton.language as tl

@triton.jit
def silu_fused_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    exp_neg_x = tl.exp(-x)
    sigmoid = 1.0 / (1.0 + exp_neg_x)
    silu_x = x * sigmoid
    out = silu_x * y

    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def silu_fused(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)
    silu_fused_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def pattern(x, y):
    tmp_0 = torch.nn.functional.silu(x, inplace=False)
    tmp_1 = tmp_0 * y
    return tmp_1

def replacement_args(x, y):
    return (x, y)

def replacement_func():
    return silu_fused