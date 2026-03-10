import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.interpolate(in_1, size=(63, 63), mode='bilinear')
    tmp_2 = tmp_1.permute(0, 2, 3, 1)
    tmp_3 = tmp_2.reshape(3969, -1)
    tmp_4 = tmp_0[slice(3969, None, None)]
    return (tmp_4, tmp_3)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(
    in_0_ptr,
    in_1_ptr,
    out_0_ptr,
    out_1_ptr,
    n_channels: tl.constexpr,
    in_0_rows: tl.constexpr,
    slice_start: tl.constexpr,
    grid_h: tl.constexpr,
    grid_w: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= grid_h * grid_w:
        return
    
    # Compute position in the spatial grid
    h = pid // grid_w
    w = pid % grid_w
    
    # Compute output sizes
    out_0_size = in_0_rows - slice_start
    out_1_size = slice_start
    
    # Process output_1: permuted and reshaped in_1 [1, 16, 63, 63] -> [1, 63, 63, 16] -> [3969, 16]
    # Each thread handles one spatial location
    if h < 63 and w < 63:
        src_idx = h * 63 * 16 + w * 16  # [1, 16, 63, 63] layout: [batch][channel][h][w]
        dst_idx = h * 63 + w  # Spatial position [3969]
        
        for i in range(n_channels):
            tl.store(out_1_ptr + dst_idx * n_channels + i, tl.load(in_1_ptr + src_idx + i))
    
    # Process output_0: sliced in_0 [slice_start:, :]
    if h == 0 and w < out_0_size:
        offset = (slice_start + w) * n_channels
        for i in range(n_channels):
            tl.store(out_0_ptr + w * n_channels + i, tl.load(in_0_ptr + offset + i))

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    in_0_shape = in_0.shape
    in_1_shape = in_1.shape
    
    n_channels = in_0_shape[1]
    in_0_rows = in_0_shape[0]
    slice_start = 3969
    
    # Calculate output sizes
    out_0_size = in_0_rows - slice_start
    out_1_size = slice_start
    
    # Create output tensors
    out_0 = torch.empty((out_0_size, n_channels), dtype=in_0.dtype, device=in_0.device)
    out_1 = torch.empty((out_1_size, n_channels), dtype=in_1.dtype, device=in_1.device)
    
    # Calculate grid dimensions for spatial processing
    grid_h = 64  # 63 for spatial, 1 for the slice operation
    grid_w = max(63, out_0_size)
    
    # Grid size for launching kernel
    grid_size = grid_h * grid_w
    
    # Launch kernel
    optimized_kernel[grid_size](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_0_ptr=out_0,
        out_1_ptr=out_1,
        n_channels=n_channels,
        in_0_rows=in_0_rows,
        slice_start=slice_start,
        grid_h=grid_h,
        grid_w=grid_w,
    )
    
    return (out_0, out_1)

def replacement_func():
    return kernel_wrapper