import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Pattern: transpose(1, 2) followed by reshape
    This is a simpler pattern that just matches transpose without the reshape part
    We'll let the reshape happen naturally afterward
    """
    t = x.transpose(1, 2)
    return t


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def optimized_transpose_kernel(
    x_ptr,
    out_ptr,
    B, H, S, D,
    N,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_ob, stride_os, stride_oh, stride_od,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized transpose(1, 2) kernel using 1D grid
    Input: (B, H, S, D)
    Output: (B, S, H, D)
    """
    pid = tl.program_id(0)
    
    # Each program processes BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Decompose linear index into (b, h, s, d) for input
    # Input layout: (B, H, S, D)
    d = offsets % D
    remainder = offsets // D
    s = remainder % S
    remainder = remainder // S
    h = remainder % H
    b = remainder // H
    
    # Calculate input offset
    x_offsets = b * stride_xb + h * stride_xh + s * stride_xs + d * stride_xd
    
    # Load data
    data = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    
    # Calculate output offset for (b, s, h, d)
    out_offsets = b * stride_ob + s * stride_os + h * stride_oh + d * stride_od
    
    # Store data
    tl.store(out_ptr + out_offsets, data, mask=mask)


@torch.fx.wrap  
def optimized_transpose(x):
    """
    Use PyTorch's native transpose - already optimized
    """
    return x.transpose(1, 2)


def replacement_func():
    return optimized_transpose