import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x):
    tmp_1 = x.norm(p = 2, dim = -1, keepdim = True)
    tmp_2 = x / tmp_1
    return tmp_2

# Argument extraction function
def replacement_args(x):
    return (x,)

# Triton kernel
@triton.jit
def norm_div_kernel(
    x_ptr,
    out_ptr,
    num_groups,
    N,
    BLOCK_SIZE: tl.constexpr
):
    group_idx = tl.program_id(0)
    block_start = group_idx * N
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (num_groups * N)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_sq = x * x
    sum_sq = tl.sum(x_sq)
    norm = tl.sqrt(sum_sq)
    out = x / norm
    out = out.to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def norm_div(x):
    N = x.shape[-1]
    num_groups = x.numel() // N
    out = torch.empty_like(x)
    BLOCK_SIZE = 512
    num_blocks = num_groups
    norm_div_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        num_groups=num_groups,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

# Replacement function
def replacement_func():
    return norm_div