import torch
import triton
import triton.language as tl



# Pattern matching function for ReLU + Addition + AdaptiveAvgPool
def pattern(in_0, in_1):
    """Match the entire computation sequence"""
    tmp_0 = torch.nn.functional.relu(in_1, inplace = False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel for the entire sequence
@triton.jit
def fused_computation_kernel(
    in_0_ptr,   # First input tensor
    in_1_ptr,   # Second input tensor (for ReLU)
    out_ptr,    # Output tensor [B, C, 1, 1]
    batch_size,
    channels,
    in_height,
    in_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output position (per channel per batch)
    pid = tl.program_id(0)
    batch_id = pid // channels
    channel_id = pid % channels
    
    # Calculate input start positions for this batch and channel
    batch_offset = batch_id * channels * in_height * in_width
    channel_offset = channel_id * in_height * in_width
    spatial_offset = batch_offset + channel_offset
    
    # Load all spatial elements for this batch and channel and perform computation
    # 1. ReLU on in_1: max(0, in_1[batch, channel, :, :])
    # 2. Add in_0: max(0, in_1) + in_0[batch, channel, :, :]
    # 3. Average across spatial dimensions
    relu_add_results = tl.empty(in_height * in_width, dtype=tl.float32)
    
    for i in range(0, in_height * in_width, BLOCK_SIZE // tl.num_programs()):
        end_idx = min(i + BLOCK_SIZE // tl.num_programs(), in_height * in_width)
        mask = (i + tl.arange(0, end_idx - i)) < (in_height * in_width)
        
        # Load elements for current block
        in_0_elements = tl.load(in_0_ptr + spatial_offset + i + tl.arange(0, end_idx - i), mask=mask, other=0.0)
        in_1_elements = tl.load(in_1_ptr + spatial_offset + i + tl.arange(0, end_idx - i), mask=mask, other=0.0)
        
        # ReLU + Addition
        relu_add = tl.maximum(in_1_elements, 0.0) + in_0_elements
        for j in range(end_idx - i):
            relu_add_results[i + j] = relu_add[j]
    
    # Compute mean across spatial dimensions
    sum_result = tl.sum(relu_add_results)
    mean_result = sum_result / (in_height * in_width)
    
    # Store result at the corresponding output position
    out_idx = batch_id * channels + channel_id
    tl.store(out_ptr + out_idx, mean_result)

@torch.fx.wrap
def fused_computation(in_0, in_1):
    """Fused entire computation: ReLU + Addition + AdaptiveAvgPool"""
    shape = in_0.shape
    batch_size, channels, in_height, in_width = shape[0], shape[1], shape[2], shape[3]
    
    # Output shape: [batch_size, channels, 1, 1]
    out_shape = (batch_size, channels, 1, 1)
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Flatten output for easier 1D kernel execution
    out_flat = out.view(-1)
    
    block_size = 1024  # Process multiple elements per thread
    total_elements = batch_size * channels
    num_programs = (total_elements + block_size - 1) // block_size
    
    if total_elements > 0:
        fused_k = fused_computation_kernel[(num_programs,)](
            in_0_ptr=in_0,
            in_1_ptr=in_1,
            out_ptr=out_flat,
            batch_size=batch_size,
            channels=channels,
            in_height=in_height,
            in_width=in_width,
            BLOCK_SIZE=block_size,
        )
    
    return out

# Replacement function
def replacement_func():
    return fused_computation