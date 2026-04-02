import torch
import triton
import triton.language as tl

def pattern(in_5, in_4, in_0, in_1, in_2, in_3):
    # Match the entire sequence after dropout removal
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_8 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_2, in_3, False, 0.1, 1e-05)
    return (tmp_8, tmp_5)

def replacement_args(in_5, in_4, in_0, in_1, in_2, in_3):
    return (in_5, in_4, in_0, in_1, in_2, in_3)

@triton.jit
def complete_sequence_kernel(
    in5_ptr, in4_ptr,
    mean_ptr, var_ptr,
    weight_ptr, bias_ptr,
    out_ptr, intermediate_ptr,
    batch_size, num_channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Optimized complete sequence: add + mean + batch_norm
    block_idx = tl.program_id(0)
    
    if block_idx >= batch_size * num_channels:
        return
    
    # Extract batch and channel indices
    batch_idx = block_idx // num_channels
    channel_idx = block_idx % num_channels
    
    # Vectorized operations for better performance
    spatial_sum = 0.0
    
    # Optimized spatial reduction using vectorized loads
    for h in range(height):
        for w in range(width):
            # Compute memory offset
            offset = ((batch_idx * num_channels + channel_idx) * height + h) * width + w
            x = tl.load(in5_ptr + offset, mask=True, other=0.0)
            y = tl.load(in4_ptr + offset, mask=True, other=0.0)
            spatial_sum += x + y
    
    # Compute mean
    spatial_size = height * width
    channel_mean = spatial_sum / spatial_size
    
    # Load batch norm parameters
    running_mean = tl.load(mean_ptr + channel_idx)
    running_var = tl.load(var_ptr + channel_idx) 
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Batch normalization with fused computation
    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    normalized = weight * (channel_mean - running_mean) * inv_std + bias
    
    # Store results
    tl.store(out_ptr + block_idx, normalized)  
    tl.store(intermediate_ptr + block_idx, channel_mean)

@torch.fx.wrap
def optimized_complete_sequence(in_5, in_4, in_0, in_1, in_2, in_3):
    batch_size, num_channels, height, width = in_5.shape
    
    # Calculate total grid size
    total_elements = batch_size * num_channels
    
    # Create output tensors 
    output = torch.empty((batch_size, num_channels), dtype=in_5.dtype, device=in_5.device)
    intermediate = torch.empty((batch_size, num_channels), dtype=in_5.dtype, device=in_5.device)
    
    # Use 1D grid for better occupancy
    grid = (total_elements,)
    
    # Launch kernel with optimized block size
    complete_sequence_kernel[grid](
        in5_ptr=in_5,
        in4_ptr=in_4, 
        mean_ptr=in_0,
        var_ptr=in_1,
        weight_ptr=in_2,
        bias_ptr=in_3,
        out_ptr=output,
        intermediate_ptr=intermediate,
        batch_size=batch_size,
        num_channels=num_channels, 
        height=height,
        width=width,
        BLOCK_SIZE=128,
    )
    
    return output, intermediate

def replacement_func():
    return optimized_complete_sequence