import torch
import triton
import triton.language as tl

def pattern(expanded_softmax_tensor, in_0):
    """
    Match the fused multiplication and sum pattern:
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1) 
    """
    tmp_4 = expanded_softmax_tensor * in_0
    result = torch.sum(tmp_4, dim=1)
    return result

def replacement_args(expanded_softmax_tensor, in_0):
    """
    Extract arguments needed for the replacement function
    """
    return (expanded_softmax_tensor, in_0)

@triton.jit
def fused_multiply_sum_kernel(
    softmax_ptr,
    in_0_ptr,
    output_ptr,
    batch_size,
    channels,
    spatial_h,
    spatial_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel that performs element-wise multiplication and channel-wise sum in one pass
    """
    # Each program handles a tile of the output tensor
    pid = tl.program_id(0)
    batch_idx = pid // (spatial_h * spatial_w)
    spatial_idx = pid % (spatial_h * spatial_w)
    h_idx = spatial_idx // spatial_w
    w_idx = spatial_idx % spatial_w
    
    # Calculate base pointers for this batch and spatial location
    softmax_base = batch_idx * channels * spatial_h * spatial_w
    in_0_base = batch_idx * channels * spatial_h * spatial_w
    output_base = batch_idx * (channels // 2) * spatial_h * spatial_w + spatial_idx
    
    # Reduce channels: sum over the channel dimension (dim=1)
    sum_val = 0.0
    for k in range(0, channels, BLOCK_SIZE_K):
        # Load softmax block (this will be [C, H, W] for each batch)
        softmax_ptr_local = softmax_ptr + softmax_base + (k + tl.arange(0, BLOCK_SIZE_K)) * spatial_h * spatial_w + h_idx * spatial_w + w_idx
        softmax_val = tl.load(softmax_ptr_local, mask=(k + tl.arange(0, BLOCK_SIZE_K)) < channels, other=0.0)
        
        # Load in_0 block (this will be [C, H, W] for each batch)
        in_0_ptr_local = in_0_ptr + in_0_base + (k + tl.arange(0, BLOCK_SIZE_K)) * spatial_h * spatial_w + h_idx * spatial_w + w_idx
        in_0_val = tl.load(in_0_ptr_local, mask=(k + tl.arange(0, BLOCK_SIZE_K)) < channels, other=0.0)
        
        # Multiply and accumulate
        sum_val += softmax_val * in_0_val
    
    # Store the summed result
    output_ptr_local = output_ptr + output_base
    tl.store(output_ptr_local, sum_val)

@torch.fx.wrap
def fused_multiply_sum(expanded_softmax_tensor, in_0):
    """
    Fused multiplication and sum operation using Triton kernel
    """
    # Get input shapes
    softmax_shape = expanded_softmax_tensor.shape
    in_0_shape = in_0.shape
    
    # Extract dimensions
    batch_size = softmax_shape[0]
    channels_softmax = softmax_shape[1]  
    spatial_h = softmax_shape[2]
    spatial_w = softmax_shape[3] if len(softmax_shape) > 3 else softmax_shape[4]
    
    # For in_0, the shape should be [batch_size, channels_in0, spatial_h, spatial_w, ...]
    channels_in0 = in_0_shape[1]
    spatial_h_in0 = in_0_shape[2]
    spatial_w_in0 = in_0_shape[3]
    
    # Check that spatial dimensions match
    assert spatial_h == spatial_h_in0, f"Spatial height mismatch: {spatial_h} vs {spatial_h_in0}"
    assert spatial_w == spatial_w_in0, f"Spatial width mismatch: {spatial_w} vs {spatial_w_in0}"
    
    # Output shape: [batch_size, channels_softmax//2, spatial_h, spatial_w] (from sum along dim=1)
    # But looking at original computation: sum over dim=1 of [B, C, H, W, 1, 1] -> [B, H, W]
    # Wait, let me recheck the original computation...

    # Original computation:
    # tmp_3: [8, 2, 128, 1, 1] * in_0: [8, 2, 128, 120, 160] -> [8, 2, 128, 120, 160]
    # tmp_5 = torch.sum(tmp_4, dim=1) -> [8, 128, 120, 160]

    # So output should be [batch_size, channels_softmax//2, spatial_h, spatial_w]
    output_shape = (batch_size, channels_softmax // 2, spatial_h, spatial_w)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=expanded_softmax_tensor.dtype, device=expanded_softmax_tensor.device)
    
    # Block sizes for tiling
    BLOCK_SIZE_M = 8   # Number of channels to process at once
    BLOCK_SIZE_K = 32  # Number of channels to reduce at once
    
    # Launch kernel - each block handles one batch and one spatial location
    num_spatial_locations = spatial_h * spatial_w
    grid_size = batch_size * num_spatial_locations
    
    # Flatten the tensors so we can process them more easily
    softmax_flat = expanded_softmax_tensor.reshape(batch_size, channels_softmax, -1)  # [B, C, H*W]
    in_0_flat = in_0.reshape(batch_size, channels_in0, -1)  # [B, C, H*W]
    
    fused_multiply_sum_kernel[grid_size](
        softmax_flat,
        in_0_flat,
        output,
        batch_size,
        channels_softmax,
        num_spatial_locations,
        spatial_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

@triton.jit
def optimized_fused_multiply_sum_kernel(
    softmax_ptr,
    in_0_ptr, 
    output_ptr,
    batch_size,
    channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    More efficient fused kernel with better memory access patterns
    """
    row = tl.program_id(0)
    col = tl.program_id(1) 
    offset = tl.program_id(2)
    
    # Handle one batch and spatial location
    spatial_idx = offset
    batch_idx = row
    
    # Each thread handles one channel, reduce over channels
    pid = tl.program_id(1)  # This gives us the channel we're responsible for
    sum_val = 0.0
    
    # Base offset for this batch and spatial location
    base_offset = batch_idx * channels * spatial_size + spatial_idx
    
    # Multiply and accumulate across the channel dimension (dim=1)
    for k in range(0, channels, BLOCK_SIZE):
        mask = (k + tl.arange(0, BLOCK_SIZE)) < channels
        
        # Load softmax and in_0 values
        softmax_offset = base_offset + (k + tl.arange(0, BLOCK_SIZE))
        softmax_vals = tl.load(softmax_ptr + softmax_offset, mask=mask, other=0.0)
        
        in_0_offset = base_offset + (k + tl.arange(0, BLOCK_SIZE)) 
        in_0_vals = tl.load(in_0_ptr + in_0_offset, mask=mask, other=0.0)
        
        sum_val += softmax_vals * in_0_vals
    
    # Store result
    output_offset = batch_idx * (channels // 2) * spatial_size + spatial_idx
    tl.store(output_ptr + output_offset, sum_val)

@torch.fx.wrap
def optimized_fused_multiply_sum(expanded_softmax_tensor, in_0):
    """
    More optimized fused multiplication and sum operation
    """
    # Get input shapes and flatten spatial dimensions
    softmax_shape = expanded_softmax_tensor.shape
    in_0_shape = in_0.shape
    
    batch_size = softmax_shape[0]
    channels_softmax = softmax_shape[1]
    spatial_total = 1
    for i in range(2, len(softmax_shape)):
        spatial_total *= softmax_shape[i]
    
    channels_in0 = in_0_shape[1]
    spatial_total_in0 = 1
    for i in range(2, len(in_0_shape)):
        spatial_total_in0 *= in_0_shape[i]
    
    assert channels_softmax == channels_in0, f"Channel mismatch: {channels_softmax} vs {channels_in0}"
    assert spatial_total == spatial_total_in0, f"Spatial size mismatch: {spatial_total} vs {spatial_total_in0}"
    
    # Reshape to [B, C, total_spatial]
    softmax_flat = expanded_softmax_tensor.reshape(batch_size, channels_softmax, spatial_total)
    in_0_flat = in_0.reshape(batch_size, channels_in0, spatial_total)
    
    # Output shape after sum over channels: [B, C//2, total_spatial]
    output_shape = (batch_size, channels_softmax // 2, spatial_total)
    output = torch.empty(output_shape, dtype=expanded_softmax_tensor.dtype, device=expanded_softmax_tensor.device)
    
    # Launch optimized kernel
    BLOCK_SIZE = 128
    num_batches = batch_size
    num_spatial = spatial_total
    grid_size = (num_batches, (channels_softmax // 2), num_spatial)
    
    optimized_fused_multiply_sum_kernel[grid_size](
        softmax_flat,
        in_0_flat, 
        output,
        batch_size,
        channels_softmax,
        spatial_total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original spatial dimensions
    # For example, if final output should be [B, C, H, W], reshape from [B, C, H*W]
    if len(softmax_shape) > 4:  # Original had 5 dimensions including ones
        final_shape = (batch_size, channels_softmax // 2) + tuple(softmax_shape[2:4])
    else:
        final_shape = (batch_size, channels_softmax // 2, spatial_total)
    
    return output.reshape(final_shape)

def replacement_func():
    """
    Return the optimized fused multiply-sum function
    """
    return optimized_fused_multiply_sum