import torch
import triton
import triton.language as tl

def pattern(tmp_1):
    """Pattern matching: adaptive_avg_pool2d with size=1"""
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2

def replacement_args(tmp_1):
    """Extract arguments for the replacement function"""
    return tmp_1,

@triton.jit
def optimized_avg_pool2d_kernel(
    input_ptr, 
    output_ptr,
    batch_size,
    channels,
    in_height,
    in_width,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for computing spatial mean (adaptive_pool2d with size=1)"""
    # Each program handles one channel in one batch
    batch = tl.program_id(0)
    channel = tl.program_id(1)
    
    # Check bounds
    if batch >= batch_size or channel >= channels:
        return
        
    # Compute spatial mean for this batch and channel
    spatial_sum = 0.0
    spatial_count = 0
    
    # Loop through spatial dimensions
    for h in range(in_height):
        for w in range(in_width):
            # Calculate input offset for current batch, channel, spatial location
            offset = (batch * channels + channel) * in_height * in_width + h * in_width + w
            
            # Load with mask and other to handle bounds safely
            val = tl.load(input_ptr + offset, mask=True, other=0.0)
            
            # Add to sum if valid (not NaN)
            if val == val:
                spatial_sum += val
                spatial_count += 1
    
    # Compute mean
    spatial_mean = spatial_sum / spatial_count if spatial_count > 0 else 0.0
    
    # Store result at output offset
    output_offset = batch * channels + channel
    tl.store(output_ptr + output_offset, spatial_mean)

@torch.fx.wrap
def optimized_adaptive_avg_pool2d(x):
    """Wrapper function for optimized adaptive_avg_pool2d with size=1"""
    batch_size, channels, height, width = x.shape
    
    # Prepare output tensor with shape [batch_size, channels, 1, 1]
    output = torch.zeros(batch_size, channels, 1, 1, device=x.device, dtype=x.dtype)
    
    # Launch kernel with grid (batch_size, channels) - each thread handles one batch-channel pair
    optimized_avg_pool2d_kernel[(
        batch_size,
        channels,
    )](
        input_ptr=x.contiguous(),
        output_ptr=output.view(batch_size, channels),
        batch_size=batch_size,
        channels=channels,
        in_height=height,
        in_width=width,
        BLOCK_SIZE=1024,
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return optimized_adaptive_avg_pool2d