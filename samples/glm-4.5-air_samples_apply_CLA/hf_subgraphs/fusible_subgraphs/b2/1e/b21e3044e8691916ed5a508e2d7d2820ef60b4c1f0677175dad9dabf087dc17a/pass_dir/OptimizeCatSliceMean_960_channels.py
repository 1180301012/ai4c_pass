import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Concatenate two tensors along channel dimension (dim=1)
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    # Slice to get first 960 channels (channel indices 0-959)
    tmp_1 = tmp_0[slice(None, None, None), slice(None, 960, None), slice(None, None, None), slice(None, None, None)]
    # Compute mean over spatial dimensions (H, W) with keepdim=True
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_cat_slice_mean_960_kernel(
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
    # Each program handles one spatial position for all batch elements
    pid_m = tl.program_id(0)  # batch
    pid_n = tl.program_id(1)  # spatial position (flattened height * width)
    
    # Calculate spatial position
    spatial_total = height * width
    spatial_id = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = spatial_id < spatial_total
    
    # Load first 480 channels from in_0 and first 480 channels from in_1
    # We need [batch, 960, height, width] -> reshape to [batch, height, width, 960]
    in0_flat = in0_ptr + pid_m * 480 * height * width + tl.arange(0, 480 * BLOCK_SIZE_N)[:, None] * (height * width) + spatial_id[None, :]
    in1_flat = in1_ptr + pid_m * 480 * height * width + tl.arange(480, 960)[:, None] * (height * width) + spatial_id[None, :]
    
    # Load both chunks - first chunk from in_0 (channels 0-479), second chunk from in_1 (channels 0-479)
    chunk0 = tl.load(in0_flat, mask=spatial_id[None, :] < spatial_total, other=0.0).to(tl.float32)
    chunk1 = tl.load(in1_flat, mask=spatial_id[None, :] < spatial_total, other=0.0).to(tl.float32)
    
    # Concatenate channels: [batch, height, width, 960]
    combined = tl.concat([chunk0, chunk1], axis=0)
    
    # Compute mean over spatial dimensions for each channel
    spatial_sum = tl.sum(combined, axis=1)
    out_mean = spatial_sum / (height * width)
    
    # Store the combined tensor and mean
    # For combined tensor: [batch, 960, height, width]
    combined_output = combined.reshape(960, BLOCK_SIZE_N)
    tl.store(out_ptr + pid_m * 960 * height * width + 
             tl.arange(0, 960)[:, None] * (height * width) + spatial_id[None, :], 
             combined_output, mask=spatial_id[None, :] < spatial_total)
    
    # Store mean: [batch, 960, 1, 1]
    tl.store(mean_ptr + pid_m * 960 + tl.arange(0, 960), out_mean)

@torch.fx.wrap
def optimized_cat_slice_mean_960(in_0, in_1):
    batch_size, channels0, height, width = in_0.shape
    assert channels0 == 480, f"Expected 480 channels, got {channels0}"
    assert in_1.shape[1] == 480, f"Expected 480 channels, got {in_1.shape[1]}"
    
    # Create output tensors
    out_slices = torch.empty((batch_size, 960, height, width), dtype=torch.float32, device=in_0.device)
    out_means = torch.empty((batch_size, 960, 1, 1), dtype=torch.float32, device=in_0.device)
    
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
    optimized_cat_slice_mean_960_kernel[grid](
        in_0,
        in_1,
        out_slices,
        out_means.flatten(),
        batch_size,
        height,
        width,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return (out_slices, out_means)

def replacement_func():
    return optimized_cat_slice_mean_960