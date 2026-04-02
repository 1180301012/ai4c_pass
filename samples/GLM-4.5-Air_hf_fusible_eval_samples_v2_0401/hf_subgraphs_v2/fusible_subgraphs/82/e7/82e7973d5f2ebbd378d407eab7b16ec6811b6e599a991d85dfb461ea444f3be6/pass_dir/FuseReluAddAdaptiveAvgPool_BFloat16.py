import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, y):
    """Match ReLU + Add + Adaptive Avg Pool pattern"""
    tmp_0 = torch.nn.functional.relu(y, inplace = False);  y = None
    tmp_1 = tmp_0 + x;  tmp_0 = x = None
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1);  tmp_1 = None
    return (tmp_2,)

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

@triton.jit
def relu_add_adaptive_avg_pool_kernel_bf16(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused ReLU + Add + Adaptive Avg Pool kernel for bfloat16"""
    
    pid = tl.program_id(0)
    
    # Split work across batch and channels
    batch_id = pid // (channels // BLOCK_SIZE_N)
    channel_group = pid % (channels // BLOCK_SIZE_N)
    channel_id = channel_group * BLOCK_SIZE_N
    
    if batch_id >= batch_size or channel_id >= channels:
        return
    
    # Compute spatial size
    spatial_size = height * width
    
    # Allocate shared memory for block processing
    x_shared = tl.static_shared_memory(BLOCK_SIZE_M * spatial_size, dtype=tl.float32)
    y_shared = tl.static_shared_memory(BLOCK_SIZE_M * spatial_size, dtype=tl.float32)
    
    # Load x and y blocks into shared memory
    for i in range(BLOCK_SIZE_M):
        current_batch = batch_id * BLOCK_SIZE_M + i
        if current_batch < batch_size:
            x_offset = (current_batch * channels + channel_id) * spatial_size
            y_offset = x_offset
            
            # Load spatial slice
            x_slice = tl.load(x_ptr + x_offset + tl.arange(0, spatial_size), 
                            mask=tl.arange(0, spatial_size) < spatial_size, 
                            other=0.0).to(tl.float32)
            y_slice = tl.load(y_ptr + y_offset + tl.arange(0, spatial_size), 
                            mask=tl.arange(0, spatial_size) < spatial_size, 
                            other=0.0).to(tl.float32)
            
            # Store in shared memory
            x_shared[i * spatial_size: (i + 1) * spatial_size] = x_slice
            y_shared[i * spatial_size: (i + 1) * spatial_size] = y_slice
    
    # Compute ReLU + Add for all batch elements in block
    relu_y = tl.maximum(y_shared, 0.0)
    sum_result = relu_y + x_shared
    
    # Adaptive average pool: compute mean for each batch element and channel group
    for i in range(BLOCK_SIZE_M):
        current_batch = batch_id * BLOCK_SIZE_M + i
        if current_batch < batch_size:
            # Process output for this batch element across the channel group
            for j in range(BLOCK_SIZE_N):
                current_channel = channel_id + j
                if current_channel < channels:
                    # Extract spatial elements for this channel
                    spatial_offset = i * spatial_size + j * spatial_size
                    spatial_elements = sum_result[spatial_offset: spatial_offset + spatial_size]
                    
                    # Compute mean
                    mean_val = tl.sum(spatial_elements) / spatial_size
                    
                    # Store result (output is [batch_size, channels, 1, 1] flattened)
                    out_idx = current_batch * channels + current_channel
                    tl.store(out_ptr + out_idx, mean_val.to(tl.bfloat16))

@torch.fx.wrap  
def fused_relu_add_adaptive_avg_pool_bf16(x, y):
    """Fused ReLU + Add + Adaptive Avg Pool wrapper for bfloat16"""
    
    # Get tensor shapes
    batch_size, channels, height, width = x.shape
    
    # Allocate output tensor [batch_size, channels, 1, 1]
    output = torch.empty((batch_size, channels, 1, 1), dtype=torch.bfloat16, device=x.device)
    
    # Flatten for processing
    x_flat = x.flatten()
    y_flat = y.flatten()
    output_flat = output.flatten()
    
    # Optimized block sizes for bfloat16 (smaller blocks for better precision)
    BLOCK_SIZE_M = min(16, batch_size)  # Process multiple batch elements
    BLOCK_SIZE_N = min(32, channels)    # Process multiple channels
    
    # Calculate grid size: batch_size * ceil(channels / BLOCK_SIZE_N)
    grid_channels = (channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_size = batch_size * grid_channels
    
    # Launch kernel
    relu_add_adaptive_avg_pool_kernel_bf16[(grid_size,)](
        x_ptr=x_flat,
        y_ptr=y_flat,
        out_ptr=output_flat,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

# Replacement function  
def replacement_func():
    return fused_relu_add_adaptive_avg_pool_bf16