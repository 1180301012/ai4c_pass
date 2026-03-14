import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Match mean reduction over spatial dimensions (2, 3) with keepdim=True"""
    result = input_tensor.mean((2, 3), keepdim=True)
    return result

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_W': 32}, num_warps=2),
    ],
    key=['H', 'W'],
)
@triton.jit
def mean_spatial_kernel(
    input_ptr,
    output_ptr,
    N, C, H, W,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """Optimized kernel for spatial mean reduction"""
    # Each program handles one (batch, channel) pair
    pid_nc = tl.program_id(0)
    n = pid_nc // C
    c = pid_nc % C
    
    # Base offset for this (N, C) slice
    base_offset = n * C * H * W + c * H * W
    
    # Accumulate sum over spatial dimensions
    sum_val = 0.0
    
    # Iterate over spatial dimensions in blocks
    for h_start in range(0, H, BLOCK_SIZE_H):
        for w_start in range(0, W, BLOCK_SIZE_W):
            h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
            w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
            
            h_mask = h_offsets < H
            w_mask = w_offsets < W
            
            # Compute 2D offsets
            h_offsets_2d = h_offsets[:, None]
            w_offsets_2d = w_offsets[None, :]
            
            offsets = base_offset + h_offsets_2d * W + w_offsets_2d
            mask = h_mask[:, None] & w_mask[None, :]
            
            # Load and accumulate
            vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
            sum_val += tl.sum(vals)
    
    # Compute mean
    spatial_size = H * W
    mean_val = sum_val / spatial_size
    
    # Store result
    output_offset = n * C + c
    tl.store(output_ptr + output_offset, mean_val)

@torch.fx.wrap
def optimized_mean_spatial(input_tensor):
    """Optimized mean reduction over spatial dimensions with keepdim=True"""
    N, C, H, W = input_tensor.shape
    
    # Output shape with keepdim=True: (N, C, 1, 1)
    output = torch.empty(N, C, 1, 1, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel: one program per (N, C) pair
    grid = (N * C,)
    
    mean_spatial_kernel[grid](
        input_tensor,
        output,
        N, C, H, W,
    )
    
    return output

def replacement_func():
    return optimized_mean_spatial