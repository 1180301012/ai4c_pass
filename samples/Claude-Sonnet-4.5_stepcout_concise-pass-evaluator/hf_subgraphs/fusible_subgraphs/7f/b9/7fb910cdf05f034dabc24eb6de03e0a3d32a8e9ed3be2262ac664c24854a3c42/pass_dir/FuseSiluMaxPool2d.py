import torch
import triton
import triton.language as tl


def pattern(in_0):
    """Pattern to match: SiLU followed by MaxPool2D with kernel_size=5, stride=1, padding=2"""
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    return (tmp_1, tmp_0)


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8}, num_warps=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=4),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 8}, num_warps=4),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 8}, num_warps=4),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 32}, num_warps=4),
    ],
    key=['H', 'W', 'C'],
)
@triton.jit
def silu_kernel(
    input_ptr,
    output_ptr,
    N, C, H, W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Optimized SiLU kernel"""
    # Get program IDs
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Compute batch and channel indices
    n = pid_nc // C
    c = pid_nc % C
    
    # Compute spatial offsets
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    
    # Create offset arrays
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    
    # Create 2D mask
    h_mask = h_offsets < H
    w_mask = w_offsets < W
    
    # Compute base offset for this NC slice
    nc_offset = (n * C + c) * H * W
    
    # Load data
    h_offsets_2d = h_offsets[:, None]
    w_offsets_2d = w_offsets[None, :]
    offsets_2d = nc_offset + h_offsets_2d * W + w_offsets_2d
    mask_2d = h_mask[:, None] & w_mask[None, :]
    
    x = tl.load(input_ptr + offsets_2d, mask=mask_2d, other=0.0)
    
    # Compute SiLU: x * sigmoid(x)
    sigmoid_x = tl.sigmoid(x)
    result = x * sigmoid_x
    
    # Store result
    tl.store(output_ptr + offsets_2d, result, mask=mask_2d)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 4}, num_warps=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8}, num_warps=4),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 8}, num_warps=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 4}, num_warps=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 4}, num_warps=4),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 16}, num_warps=4),
    ],
    key=['H', 'W', 'C'],
)
@triton.jit
def maxpool2d_kernel(
    input_ptr,
    output_ptr,
    N, C, H, W,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Optimized MaxPool2D kernel for kernel_size=5, stride=1, padding=2, dilation=1"""
    # Get program IDs
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Compute batch and channel indices
    n = pid_nc // C
    c = pid_nc % C
    
    # Compute output spatial offsets
    h_out_start = pid_h * BLOCK_H
    w_out_start = pid_w * BLOCK_W
    
    # Create offset arrays for output
    h_out_offsets = h_out_start + tl.arange(0, BLOCK_H)
    w_out_offsets = w_out_start + tl.arange(0, BLOCK_W)
    
    # Create 2D mask for output
    h_out_mask = h_out_offsets < H
    w_out_mask = w_out_offsets < W
    mask_out = h_out_mask[:, None] & w_out_mask[None, :]
    
    # Compute base offset for this NC slice
    nc_offset = (n * C + c) * H * W
    
    # Initialize max values with -inf
    max_vals = tl.full((BLOCK_H, BLOCK_W), float('-inf'), dtype=tl.float32)
    
    # Iterate over kernel window
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Compute input coordinates
            h_in = h_out_offsets[:, None] * stride + kh * dilation - padding
            w_in = w_out_offsets[None, :] * stride + kw * dilation - padding
            
            # Check bounds
            h_valid = (h_in >= 0) & (h_in < H)
            w_valid = (w_in >= 0) & (w_in < W)
            mask_in = h_valid & w_valid & mask_out
            
            # Load values
            offsets_in = nc_offset + h_in * W + w_in
            vals = tl.load(input_ptr + offsets_in, mask=mask_in, other=float('-inf'))
            
            # Update max
            max_vals = tl.maximum(max_vals, vals)
    
    # Store result
    offsets_out = nc_offset + h_out_offsets[:, None] * W + w_out_offsets[None, :]
    tl.store(output_ptr + offsets_out, max_vals, mask=mask_out)


@torch.fx.wrap
def fused_silu_maxpool2d(x):
    """Wrapper for fused SiLU + MaxPool2D"""
    N, C, H, W = x.shape
    
    # Allocate outputs
    silu_out = torch.empty_like(x)
    pool_out = torch.empty_like(x)
    
    # Launch SiLU kernel
    grid_silu = lambda META: (
        N * C,
        triton.cdiv(H, META['BLOCK_H']),
        triton.cdiv(W, META['BLOCK_W']),
    )
    silu_kernel[grid_silu](
        x, silu_out,
        N, C, H, W,
    )
    
    # Launch MaxPool2D kernel
    grid_pool = lambda META: (
        N * C,
        triton.cdiv(H, META['BLOCK_H']),
        triton.cdiv(W, META['BLOCK_W']),
    )
    maxpool2d_kernel[grid_pool](
        silu_out, pool_out,
        N, C, H, W,
        kernel_size=5,
        stride=1,
        padding=2,
        dilation=1,
    )
    
    return pool_out, silu_out


def replacement_func():
    return fused_silu_maxpool2d