import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['n_samples', 'c_out', 'h_in', 'w_in'],
)
@triton.jit
def fused_sum_adaptive_avg_pool_kernel(
    input_ptr,
    output_ptr,
    n_samples,
    c_out,
    h_in,
    w_in,
    BLOCK_SIZE: tl.constexpr,
):
    # Program handles one sample
    sample_id = tl.program_id(0)
    if sample_id >= n_samples:
        return
    
    # Vectorized spatial processing for one channel
    spatial_offset = tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offset < (h_in * w_in)
    
    # Process channels in a simple loop
    for c in range(c_out):
        # Initialize sum for this channel (reduction across all spatial locations)
        channel_sum = 0.0
        
        # Process spatial locations in blocks for better memory access
        spatial_blocks = (h_in * w_in + BLOCK_SIZE - 1) // BLOCK_SIZE
        for block in range(spatial_blocks):
            block_start = block * BLOCK_SIZE
            block_end = min(block_start + BLOCK_SIZE, h_in * w_in)
            
            spatial_idx = block_start + spatial_offset
            valid_mask = spatial_idx < block_end
            
            # Load pair from dim=1 for this channel and spatial block
            base_idx = (sample_id * 2 * c_out * h_in * w_in + 
                       c * h_in * w_in * 2 + 
                       spatial_idx * 2)
            
            # Load both elements (vectorized across spatial dimension)
            # Masked loads will automatically handle boundaries
            val1 = tl.load(input_ptr + base_idx, mask=valid_mask, other=0.0)
            val2 = tl.load(input_ptr + base_idx + 1, mask=valid_mask, other=0.0)
            
            # Add to channel sum (masked sum will only sum valid elements)
            pair_sum = val1 + val2
            channel_sum += tl.sum(pair_sum)
        
        # Compute average for this channel
        spatial_total = h_in * w_in
        if spatial_total > 0:
            channel_avg = channel_sum / (2 * spatial_total)
        else:
            channel_avg = 0.0
        
        # Store result for this channel
        output_offset = sample_id * c_out + c
        tl.store(output_ptr + output_offset, channel_avg)

def pattern(in_0):
    tmp_0 = in_0.sum(dim=1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

@torch.fx.wrap
def fused_sum_adaptive_avg_pool(in_0):
    # Get input shape
    input_shape = in_0.shape
    n_samples = input_shape[0]
    c_out = input_shape[2]  # After summing dim=1, this becomes the channel dim
    h_in = input_shape[3]
    w_in = input_shape[4]
    
    # Output shape: [n_samples, c_out, 1, 1]
    output_shape = [n_samples, c_out, 1, 1]
    
    # Calculate number of elements
    total_output_elements = n_samples * c_out
    
    # Create output tensor
    output = torch.empty(total_output_elements, dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel with optimized block size based on spatial dimensions
    spatial_elements = h_in * w_in
    if spatial_elements <= 64:
        block_size = 64
    elif spatial_elements <= 256:
        block_size = 128
    else:
        block_size = 256
    
    # Grid size: one program per sample
    grid_size = n_samples
    
    fused_sum_adaptive_avg_pool_kernel[(grid_size,)](
        in_0,
        output,
        n_samples,
        c_out,
        h_in,
        w_in,
        block_size,
    )
    
    # Reshape output to [n_samples, c_out, 1, 1]
    return output.reshape(output_shape)

def replacement_func():
    return fused_sum_adaptive_avg_pool