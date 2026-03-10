import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Concatenate two tensors along channel dimension (dim=1)
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    # Slice to get first 120 channels (channel indices 0-119)
    tmp_1 = tmp_0[slice(None, None, None), slice(None, 120, None), slice(None, None, None), slice(None, None, None)]
    # Compute mean over spatial dimensions (H, W) with keepdim=True
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_cat_slice_mean_120_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    mean_ptr,
    batch_size,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one batch element and spatial position
    pid_m = tl.program_id(0)  # batch
    pid_n = tl.program_id(1)  # spatial position (flattened height * width)
    
    # Calculate spatial position
    spatial_id = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    spatial_total = height * width
    mask = spatial_id < spatial_total
    
    # Process each channel separately (0-119)
    for chan_idx in range(0, 120):
        # Check if this channel exists in input_0 or input_1
        if chan_idx < 60:
            # Channel from input_0 (first 60 channels)
            in_ptr = in0_ptr + pid_m * 60 * height * width + chan_idx * height * width + spatial_id
        else:
            # Channel from input_1 (next 60 channels)
            in_ptr = in1_ptr + pid_m * 60 * height * width + (chan_idx - 60) * height * width + spatial_id
        
        # Load input data
        in_data = tl.load(in_ptr, mask=spatial_id < spatial_total, other=0.0).to(tl.float32)
        
        # Store to output tensor: [batch, 120, height, width]
        out_ptr_offset = pid_m * 120 * height * width + chan_idx * height * width + spatial_id
        tl.store(out_ptr + out_ptr_offset, in_data, mask=spatial_id < spatial_total)
        
        # Accumulate sum for mean calculation
        if chan_idx == 0:
            spatial_sum = in_data
        else:
            spatial_sum += in_data
    
    # Calculate and store mean
    out_mean = spatial_sum / (height * width)
    mean_offset = pid_m * 120 + tl.arange(0, 120)
    tl.store(mean_ptr + mean_offset, out_mean)

@torch.fx.wrap
def optimized_cat_slice_mean_120(in_0, in_1):
    batch_size, channels0, height, width = in_0.shape
    assert channels0 == 60, f"Expected 60 channels, got {channels0}"
    assert in_1.shape[1] == 60, f"Expected 60 channels, got {in_1.shape[1]}"
    
    # Create output tensors
    out_slices = torch.empty((batch_size, 120, height, width), dtype=torch.float32, device=in_0.device)
    out_means = torch.empty((batch_size, 120), dtype=torch.float32, device=in_0.device)
    
    # Set up kernel launch parameters
    BLOCK_SIZE_M = 1  # Process one batch element per program
    BLOCK_SIZE_N = 32  # Process multiple spatial positions per program
    
    # Calculate grid dimensions
    spatial_total = height * width
    grid = (
        batch_size,
        (spatial_total + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    )
    
    # Launch kernel
    optimized_cat_slice_mean_120_kernel[grid](
        in_0,
        in_1,
        out_slices,
        out_means,
        batch_size,
        height,
        width,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return (out_slices, out_means.unsqueeze(-1).unsqueeze(-1))

def replacement_func():
    return optimized_cat_slice_mean_120