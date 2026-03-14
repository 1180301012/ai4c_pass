import torch
import triton
import triton.language as tl


def pattern(tmp_4):
    """
    Pattern to match: sigmoid → subtract → multiply
    """
    tmp_5 = tmp_4.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7


def replacement_args(tmp_4):
    return (tmp_4,)


@triton.jit
def fused_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused sigmoid-sub-mul kernel
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    y = (tl.sigmoid(x) - 0.25) * 3.141592653589793
    tl.store(out_ptr + offs, y, mask=mask)


@torch.fx.wrap
def fused_sigmoid_sub_mul(tmp_4):
    """
    Fused implementation
    """
    n = tmp_4.numel()
    out = torch.empty_like(tmp_4)
    
    # Try to use optimal block size based on tensor size
    if n < 10000:
        BLOCK_SIZE = 512
    elif n < 100000:
        BLOCK_SIZE = 1024  
    else:
        BLOCK_SIZE = 2048
    
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_kernel[grid](tmp_4, out, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


def replacement_func():
    return fused_sigmoid_sub_mul