import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    """
    Pattern: ReLU → adaptive_avg_pool2d(kernel=1) → flatten(1, -1)
    Since adaptive_avg_pool2d with kernel=1 is equivalent to global average pooling
    """
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7

def replacement_args(tmp_4):
    return (tmp_4,)

@triton.jit
def fused_relu_global_pool_kernel(
    x_ptr,
    out_ptr,
    batch,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch * channels  # Output is [batch, channels] after global pooling
    
    # Each program handles a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Initialize sum and count for each element in output
    sum_val = tl.zeros((1,), dtype=tl.float32)
    count = 0.0
    
    # For each output element (batch, channel), sum over all spatial positions
    for h_idx in range(height):
        for w_idx in range(width):
            # Calculate global offset for this spatial position
            spatial_offsets = offsets + (h_idx * width + w_idx) * batch * channels
            
            # Load input values for this spatial position
            x = tl.load(x_ptr + spatial_offsets, mask=spatial_offsets < batch * channels * height * width, other=0.0)
            
            # Apply ReLU and accumulate
            relu_x = tl.maximum(x, 0.0)
            sum_val += relu_x.flatten()
            count += 1.0
    
    # Compute average
    mean_val = sum_val / count
    
    # Store result
    tl.store(out_ptr + offsets, mean_val, mask=mask)

@torch.fx.wrap  
def fused_relu_global_pool(tmp_4):
    """Fused ReLU and global average pooling"""
    input_shape = tmp_4.shape
    batch, channels, height, width = input_shape
    
    # Create output tensor [batch, channels]
    out = torch.empty((batch, channels), dtype=tmp_4.dtype, device=tmp_4.device)
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size based on output size
    grid_size = ((batch * channels) + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_relu_global_pool_kernel[grid_size](
        tmp_4,
        out,
        batch,
        channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_relu_global_pool