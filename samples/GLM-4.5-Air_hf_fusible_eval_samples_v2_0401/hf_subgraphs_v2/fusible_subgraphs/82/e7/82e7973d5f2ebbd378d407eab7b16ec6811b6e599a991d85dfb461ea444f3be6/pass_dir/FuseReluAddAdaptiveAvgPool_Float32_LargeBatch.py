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
def relu_add_adaptive_avg_pool_kernel_fp32_large_batch(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused ReLU + Add + Adaptive Avg Pool kernel for float32 large batches"""
    
    pid = tl.program_id(0)
    
    # Split work across batch and channels
    batch_id = pid // (channels + BLOCK_M - 1) // BLOCK_M
    channel_id = (pid % ((channels + BLOCK_N - 1) // BLOCK_N)) * BLOCK_N
    
    if batch_id >= batch_size or channel_id >= channels:
        return
    
    # Load input elements for this batch and channel combination
    x_offset = (batch_id * channels + channel_id) * height * width
    y_offset = x_offset
    
    # Load x and y slices for this batch-channel combination
    x_block = tl.load(x_ptr + x_offset + tl.arange(0, BLOCK_M * height * width), 
                     mask=tl.arange(0, BLOCK_M * height * width) < BLOCK_M * height * width, 
                     other=0.0)
    y_block = tl.load(y_ptr + y_offset + tl.arange(0, BLOCK_M * height * width), 
                     mask=tl.arange(0, BLOCK_M * height * width) < BLOCK_M * height * width, 
                     other=0.0)
    
    # Reshape for processing [BLOCK_M, height, width]
    x_reshaped = x_block.reshape((BLOCK_M, height, width))
    y_reshaped = y_block.reshape((BLOCK_M, height, width))
    
    # Compute ReLU on y and add x for all elements in the block
    relu_y = tl.maximum(y_reshaped, 0.0)
    sum_result = relu_y + x_reshaped
    
    # Adaptive average pool to 1x1 - compute mean for each channel slice
    spatial_size = height * width
    
    # Process each batch element in the block
    for i in range(BLOCK_M):
        if batch_id * BLOCK_M + i < batch_size:
            # Compute mean for this batch element and channel
            mean_val = tl.sum(sum_result[i]) / spatial_size
            
            # Store result (output is [batch_size, channels, 1, 1] flattened)
            out_idx = (batch_id * BLOCK_M + i) * channels + channel_id
            tl.store(out_ptr + out_idx, mean_val)

@torch.fx.wrap
def fused_relu_add_adaptive_avg_pool_fp32_large_batch(x, y):
    """Fused ReLU + Add + Adaptive Avg Pool wrapper for float32 large batches"""
    
    # Get tensor shapes
    batch_size, channels, height, width = x.shape
    
    # Allocate output tensor [batch_size, channels, 1, 1]
    output = torch.empty((batch_size, channels, 1, 1), dtype=torch.float32, device=x.device)
    
    # Flatten for processing
    x_flat = x.flatten()
    y_flat = y.flatten()
    output_flat = output.flatten()
    
    # Tunable block sizes
    BLOCK_M = min(32, batch_size)  # Process multiple batch elements
    BLOCK_N = min(64, channels)     # Process multiple channels
    
    # Calculate grid size: ceil(batch_size / BLOCK_M) * ceil(channels / BLOCK_N)
    grid_m = (batch_size + BLOCK_M - 1) // BLOCK_M
    grid_n = (channels + BLOCK_N - 1) // BLOCK_N
    grid_size = grid_m * grid_n
    
    # Launch kernel
    relu_add_adaptive_avg_pool_kernel_fp32_large_batch[(grid_size,)](
        x_ptr=x_flat,
        y_ptr=y_flat,
        out_ptr=output_flat,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output.reshape(batch_size, channels, 1, 1)

# Replacement function
def replacement_func():
    return fused_relu_add_adaptive_avg_pool_fp32_large_batch