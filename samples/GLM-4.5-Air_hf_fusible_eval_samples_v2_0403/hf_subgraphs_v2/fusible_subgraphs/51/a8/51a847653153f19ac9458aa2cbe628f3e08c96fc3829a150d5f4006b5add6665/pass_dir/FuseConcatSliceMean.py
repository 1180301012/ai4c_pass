import torch
import triton
import triton.language as tl

# Pattern matching function - matches the computation exactly for slice value 672
def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    slice_spec = (slice(None, None, None), slice(None, 672, None), slice(None, None, None), slice(None, None, None))
    tmp_1 = tmp_0[slice_spec] 
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2

# Argument extraction function - extracts inputs needed for replacement
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel that fuses concatenation, slicing, and mean computation
@triton.jit
def fused_concat_slice_mean_kernel(
    in0_ptr, in1_ptr, 
    sliced_out_ptr, mean_out_ptr,
    n_batch, n_channels_in0, n_channels_in1, height, width, target_channels,
    BLOCK_SIZE: tl.constexpr,
    SPATIAL_BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate which batch this program handles
    batch_offset = pid * SPATIAL_BLOCK_SIZE
    batch_mask = batch_offset < n_batch
    
    if not batch_mask:
        return
    
    # Compute slice parameters
    channels_from_in0 = min(target_channels, n_channels_in0)
    channels_from_in1 = max(0, target_channels - channels_from_in0)
    
    # Launch spatial kernel for this batch
    fused_slice_mean_kernel[
        (height, width), (SPATIAL_BLOCK_SIZE, SPATIAL_BLOCK_SIZE)
    ](
        in0_ptr, in1_ptr, sliced_out_ptr, mean_out_ptr,
        batch_offset, n_batch, n_channels_in0, n_channels_in1, height, width,
        channels_from_in0, channels_from_in1, target_channels,
        BLOCK_SIZE
    )

@triton.jit
def fused_slice_mean_kernel(
    in0_ptr, in1_ptr, sliced_out_ptr, mean_out_ptr,
    batch_idx, n_batch, n_channels_in0, n_channels_in1, height, width,
    channels_from_in0, channels_from_in1, target_channels,
    BLOCK_SIZE: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Calculate spatial coordinates
    h = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    w = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create coordinate grids
    h_grid = h[:, None]
    w_grid = w[None, :]
    
    # Create masks for spatial dimensions
    mask_h = h_grid < height
    mask_w = w_grid < width
    mask = mask_h & mask_w
    
    # Initialize accumulators
    sum_val = tl.zeros([target_channels], dtype=tl.float32)
    count = tl.zeros([target_channels], dtype=tl.int32)
    
    # Process each target channel
    for c in range(target_channels):
        # Determine which input this channel comes from
        if c < channels_from_in0:
            # Channel comes from in_0
            src_channel = c
            src_ptr = in0_ptr
            src_n_channels = n_channels_in0
        else:
            # Channel comes from in_1
            src_channel = c - channels_from_in0
            src_ptr = in1_ptr
            src_n_channels = n_channels_in1
        
        if src_channel >= src_n_channels:
            continue  # Skip if source channel doesn't exist
        
        # Load input tensor elements for this channel
        in0_ptr_batch = src_ptr + batch_idx * src_n_channels * height * width + src_channel * height * width
        in_vals = tl.load(in0_ptr_batch + h_grid * width + w_grid, mask=mask, other=0.0)
        
        # Store to sliced output
        out_ptr_batch = sliced_out_ptr + batch_idx * target_channels * height * width + c * height * width
        tl.store(out_ptr_batch + h_grid * width + w_grid, in_vals, mask=mask)
        
        # Accumulate for mean computation
        sum_val[c] += tl.sum(in_vals)
        count[c] += tl.sum(mask.to(tl.int32))
    
    # Compute mean and update global mean buffer
    batch_mean_offset = mean_out_ptr + batch_idx * target_channels
    mean_vals = sum_val / (count + 1e-6)  # Add small epsilon to avoid division by zero
    
    # Store mean values
    tl.store(batch_mean_offset + tl.arange(0, target_channels), mean_vals)

@torch.fx.wrap
def fused_concat_slice_mean(in_0, in_1):
    batch, channels_in0, height, width = in_0.shape
    channels_in1 = in_1.shape[1]
    target_channels = 672
    
    # Validate input shapes
    assert in_0.shape[0] == in_1.shape[0], "Batch dimensions must match"
    assert in_0.shape[2] == in_1.shape[2], "Height dimensions must match"
    assert in_0.shape[3] == in_1.shape[3], "Width dimensions must match"
    
    # Determine how many channels from each input
    channels_from_in0 = min(target_channels, channels_in0)
    channels_from_in1 = max(0, target_channels - channels_from_in0)
    
    # Create output tensors
    sliced_output = torch.empty((batch, target_channels, height, width), dtype=in_0.dtype, device=in_0.device)
    mean_output = torch.empty((batch, target_channels, 1, 1), dtype=torch.float32, device=in_0.device)
    
    # Launch kernel
    fused_concat_slice_mean_kernel[(batch,)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        sliced_out_ptr=sliced_output,
        mean_out_ptr=mean_output,
        n_batch=batch,
        n_channels_in0=channels_in0,
        n_channels_in1=channels_in1,
        height=height,
        width=width,
        target_channels=target_channels,
        BLOCK_SIZE=16,  # Tile size for spatial dimensions
        SPATIAL_BLOCK_SIZE=1  # One batch per program for simplicity
    )
    
    return sliced_output, mean_output

# Replacement function - returns the optimized kernel function
def replacement_func():
    return fused_concat_slice_mean