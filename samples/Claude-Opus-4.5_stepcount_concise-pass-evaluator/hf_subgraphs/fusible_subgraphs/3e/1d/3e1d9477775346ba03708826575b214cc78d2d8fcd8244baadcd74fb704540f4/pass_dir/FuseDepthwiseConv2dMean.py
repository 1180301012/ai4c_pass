import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Pattern to match mean over spatial dimensions (2, 3) with keepdim=True
    """
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


@triton.jit
def mean_kernel_2048(
    input_ptr,
    output_ptr,
    HW,
):
    """Compute mean over spatial dimensions with BLOCK_SIZE=2048"""
    BLOCK_SIZE: tl.constexpr = 2048
    pid = tl.program_id(0)
    base_offset = pid * HW
    
    acc = 0.0
    for start in range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        idx = base_offset + offs
        vals = tl.load(input_ptr + idx, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    
    mean_val = acc / HW
    tl.store(output_ptr + pid, mean_val)


@torch.fx.wrap
def fast_spatial_mean(x):
    """Fast spatial mean using Triton"""
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    
    out = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)
    
    grid = (N * C,)
    mean_kernel_2048[grid](x, out, HW, num_warps=4)
    
    return out


def replacement_func():
    return fast_spatial_mean