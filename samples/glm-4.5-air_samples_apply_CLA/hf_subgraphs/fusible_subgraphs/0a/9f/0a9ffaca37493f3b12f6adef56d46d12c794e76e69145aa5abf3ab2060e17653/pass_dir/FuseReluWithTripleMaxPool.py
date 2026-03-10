import torch
import triton
import triton.language as tl

# Pattern matching function: ReLU + triple max_pool2d operations
def pattern(input_tensor):
    # Apply ReLU (exactly as in model)
    relu_out = torch.nn.functional.relu(input_tensor)
    
    # Apply three identical max_pool2d operations (exactly as in model)
    pool1 = torch.nn.functional.max_pool2d(relu_out, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    pool2 = torch.nn.functional.max_pool2d(relu_out, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    pool3 = torch.nn.functional.max_pool2d(relu_out, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    
    # Return the tensors that would be used for concatenation
    return relu_out, pool1, pool2, pool3

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Triton kernel for fused ReLU + Triple MaxPool2D
@triton.jit
def fused_relu_triple_maxpool_kernel(
    input_ptr,
    relu_out_ptr,
    pool1_ptr,
    pool2_ptr,
    pool3_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate batch and channel indices
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    if batch_idx >= batch_size or channel_idx >= channels:
        return
    
    # Calculate block dimensions
    x_base = channel_idx * height * width
    y_base = batch_idx * height * width
    
    # Calculate global spatial position - each program handles one element
    spatial_size = height * width
    spatial_idx = (pid % (channels * spatial_size))
    channel_idx = spatial_idx // spatial_size
    spatial_idx = spatial_idx % spatial_size
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    if batch_idx >= batch_size or channel_idx >= channels or h_idx >= height or w_idx >= width:
        return
    
    # Load input value at this position for ReLU
    input_offset = (batch_idx * channels + channel_idx) * height * width + h_idx * width + w_idx
    input_val = tl.load(input_ptr + input_offset)
    
    # Apply ReLU and store
    relu_val = tl.maximum(input_val, 0.0)
    tl.store(relu_out_ptr + input_offset, relu_val)
    
    # Compute max pooling for all three operations with stride=1, padding=2, kernel_size=5
    for pool_idx in range(3):
        # Calculate pooling window boundaries with padding
        start_h = max(0, h_idx - 2)
        end_h = min(height, h_idx + 3)
        start_w = max(0, w_idx - 2)
        end_w = min(width, w_idx + 3)
        
        # Find max in the 5x5 pooling window (after ReLU)
        max_val = -float('inf')
        for ph in range(start_h, end_h):
            for pw in range(start_w, end_w):
                window_offset = (batch_idx * channels + channel_idx) * height * width + ph * width + pw
                window_val = tl.load(input_ptr + window_offset)
                relu_window_val = tl.maximum(window_val, 0.0)
                if relu_window_val > max_val:
                    max_val = relu_window_val
        
        # Store max pooling result
        pool_offset = (batch_idx * channels + channel_idx) * height * width + h_idx * width + w_idx
        if pool_idx == 0:
            tl.store(pool1_ptr + pool_offset, max_val)
        elif pool_idx == 1:
            tl.store(pool2_ptr + pool_offset, max_val)
        else:
            tl.store(pool3_ptr + pool_offset, max_val)

@triton.jit
def simple_relu_maxpool_kernel(
    input_ptr,
    relu_out_ptr,
    pool_out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate batch and channel indices
    batch_idx = pid // (channels * height * width)
    spatial_idx = pid % (channels * height * width)
    channel_idx = spatial_idx // (height * width)
    spatial_idx = spatial_idx % (height * width)
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    if batch_idx >= batch_size or channel_idx >= channels or h_idx >= height or w_idx >= width:
        return
    
    # Load input value at this position
    input_offset = (batch_idx * channels + channel_idx) * height * width + h_idx * width + w_idx
    input_val = tl.load(input_ptr + input_offset)
    
    # Apply ReLU and store
    relu_val = tl.maximum(input_val, 0.0)
    tl.store(relu_out_ptr + input_offset, relu_val)
    
    # Simple max pooling with stride=1, padding=2, kernel_size=5
    start_h = max(0, h_idx - 2)
    end_h = min(height, h_idx + 3)
    start_w = max(0, w_idx - 2)
    end_w = min(width, w_idx + 3)
    
    # Find max in the pooling window
    max_val = -float('inf')
    for ph in range(start_h, end_h):
        for pw in range(start_w, end_w):
            window_offset = (batch_idx * channels + channel_idx) * height * width + ph * width + pw
            window_val = tl.load(input_ptr + window_offset)
            relu_window_val = tl.maximum(window_val, 0.0)
            if relu_window_val > max_val:
                max_val = relu_window_val
    
    # Store max pooling result
    tl.store(pool_out_ptr + input_offset, max_val)

@triton.jit
def fused_relu_triple_maxpool_kernel(
    input_ptr,
    relu_out_ptr,
    pool1_ptr,
    pool2_ptr,
    pool3_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate batch and channel indices
    batch_idx = pid // (channels * height * width)
    spatial_idx = pid % (channels * height * width)
    channel_idx = spatial_idx // (height * width)
    spatial_idx = spatial_idx % (height * width)
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    if batch_idx >= batch_size or channel_idx >= channels or h_idx >= height or w_idx >= width:
        return
    
    # Load input value at this position
    input_offset = (batch_idx * channels + channel_idx) * height * width + h_idx * width + w_idx
    input_val = tl.load(input_ptr + input_offset)
    
    # Apply ReLU and store
    relu_val = tl.maximum(input_val, 0.0)
    tl.store(relu_out_ptr + input_offset, relu_val)
    
    # Compute max pooling for all three operations with stride=1, padding=2, kernel_size=5
    start_h = max(0, h_idx - 2)
    end_h = min(height, h_idx + 3)
    start_w = max(0, w_idx - 2)
    end_w = min(width, w_idx + 3)
    
    # Find max in the 5x5 pooling window (after ReLU)
    max_val = -float('inf')
    for ph in range(start_h, end_h):
        for pw in range(start_w, end_w):
            window_offset = (batch_idx * channels + channel_idx) * height * width + ph * width + pw
            window_val = tl.load(input_ptr + window_offset)
            relu_window_val = tl.maximum(window_val, 0.0)
            if relu_window_val > max_val:
                max_val = relu_window_val
    
    # Store max pooling result for all three operations
    pool_offset = (batch_idx * channels + channel_idx) * height * width + h_idx * width + w_idx
    tl.store(pool1_ptr + pool_offset, max_val)
    tl.store(pool2_ptr + pool_offset, max_val)
    tl.store(pool3_ptr + pool_offset, max_val)

@torch.fx.wrap
def fused_relu_triple_maxpool(input_tensor):
    # Get input dimensions
    batch_size, channels, height, width = input_tensor.shape
    
    # Create output tensors
    relu_out = torch.empty_like(input_tensor)
    pool1 = torch.empty_like(input_tensor)
    pool2 = torch.empty_like(input_tensor)
    pool3 = torch.empty_like(input_tensor)
    
    # Calculate grid size
    total_elements = batch_size * channels * height * width
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    
    grid = (triton.cdiv(total_elements, BLOCK_SIZE_X * BLOCK_SIZE_Y),)
    
    # Launch kernel
    fused_relu_triple_maxpool_kernel[grid](
        input_tensor,
        relu_out,
        pool1,
        pool2,
        pool3,
        batch_size,
        channels,
        height,
        width,
        BLOCK_SIZE_X,
        BLOCK_SIZE_Y,
    )
    
    return relu_out, pool1, pool2, pool3

# Replacement function - return actual implementation
def replacement_func():
    return fused_relu_triple_maxpool