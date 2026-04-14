import torch
import triton
import triton.language as tl


# Pattern matching function - must exactly mirror model.py
def pattern(in_0, in_1):
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    n_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Fused kernel that does:
    1. Element-wise multiplication: x * y
    2. Sum along channel dimension (dim=1) 
    3. Implicit unsqueeze at position 1 by output layout
    4. Sigmoid activation
    
    Input shapes: [batch_size, n_channels, height, width]
    Output shape: [batch_size, 1, height, width]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Each program handles one output element [batch, 1, height, width]
    batch_idx = pid_m
    spatial_idx = pid_n
    
    # Convert spatial index to height and width coordinates
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Process channels in blocks for better memory coalescing
    sum_val = 0.0
    
    # Load channel block loop
    for c_start in range(0, n_channels, BLOCK_SIZE_C):
        # Calculate current channel block bounds
        c_end = min(c_start + BLOCK_SIZE_C, n_channels)
        
        # Calculate base offset for this channel block and spatial location
        base_offset = (batch_idx * n_channels * height * width + 
                      c_start * height * width + 
                      h_idx * width + 
                      w_idx).to(tl.int64)
        
        # Load channel block values with bounds checking
        channel_offs = tl.arange(0, BLOCK_SIZE_C)
        mask = channel_offs < (c_end - c_start)
        
        x_vals = tl.load(x_ptr + base_offset + channel_offs, mask=mask, other=0.0)
        y_vals = tl.load(y_ptr + base_offset + channel_offs, mask=mask, other=0.0)
        
        # Vectorized multiply and accumulate
        products = x_vals * y_vals
        sum_val += tl.sum(products)
    
    # Apply sigmoid using fused operations for better performance
    exp_val = tl.exp(-sum_val)
    sigmoid_val = 1.0 / (1.0 + exp_val)
    
    # Store result at output[batch_idx, 0, h_idx, w_idx]
    out_idx = (batch_idx * height * width + spatial_idx).to(tl.int64)
    tl.store(out_ptr + out_idx, sigmoid_val)


@torch.fx.wrap
def fused_mult_sum_sigmoid(x, y):
    """
    Optimized fused operation: element-wise multiply + sum along dim=1 + sigmoid
    """
    # Get input shapes: [batch_size, n_channels, height, width]
    batch_size, n_channels, height, width = x.shape
    
    # Create output tensor with same dtype as inputs for consistency
    output = torch.empty((batch_size, 1, height, width), dtype=x.dtype, device=x.device)
    
    # Set up grid dimensions
    # grid_m: batch_size, grid_n: height * width (one program per output element)
    grid_m = batch_size
    grid_n = height * width
    
    # Use optimal block sizes for better performance
    BLOCK_SIZE_M = 32  # Batch processing block size
    BLOCK_SIZE_C = min(256, n_channels)  # Channel processing block size (adaptive)
    
    # Launch kernel - work with original tensor shapes
    fused_kernel[(grid_m, grid_n)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=output,
        batch_size=batch_size,
        n_channels=n_channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return output


# Replacement function (returns function reference, not a call)
def replacement_func():
    return fused_mult_sum_sigmoid