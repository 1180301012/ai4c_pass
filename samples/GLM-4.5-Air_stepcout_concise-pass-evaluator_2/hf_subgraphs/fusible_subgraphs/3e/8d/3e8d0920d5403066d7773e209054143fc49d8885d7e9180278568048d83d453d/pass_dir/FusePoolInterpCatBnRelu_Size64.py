import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_pool_interp_cat_bn_relu_kernel_64(
    # Input tensors
    input_ptr,                  # Input tensor from max_pool
    concat_tensor_ptr,          # Tensor to concatenate with
    running_mean_ptr,           # Batch norm running mean
    running_var_ptr,            # Batch norm running variance  
    weight_ptr,                 # Batch norm weight
    bias_ptr,                   # Batch norm bias
    
    # Output tensor
    output_ptr,
    
    # Shape information
    batch_size: tl.constexpr,
    num_channels_concat: tl.constexpr,
    num_channels_input: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    
    # Parameters
    epsilon: tl.constexpr,
    momentum: tl.constexpr,
    
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ids for 2D grid
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Compute output coordinates for 64x64 target
    out_h = pid_h * BLOCK_SIZE_M
    out_w = pid_w * BLOCK_SIZE_N
    out_c = pid_m * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Mask to handle boundary conditions
    mask_h = out_h < 64
    mask_w = out_w < 64
    mask_c = out_c < (num_channels_input + num_channels_concat)
    mask = mask_h & mask_w & mask_c
    
    # Load concatenated tensor data (for the first num_channels_concat channels)
    if out_c < num_channels_concat:
        # Load from concatenation tensor at position out_c
        concat_val = tl.load(concat_tensor_ptr + out_c * 64 * 64 + out_h * 64 + out_w, mask=mask, other=0.0)
        bn_input = concat_val
    else:
        # Load from max_pool result (shifted by concat size)
        pool_c = out_c - num_channels_concat
        
        # 2x2 max pooling with stride 2 from 128x128 to 64x64
        src_h = out_h * 2  # 0->0,1->2,2->4...
        src_w = out_w * 2  # 0->0,1->2,2->4...
        
        # For 2x2 max pooling, we need to find max in 2x2 neighborhood
        max_val = tl.minimum(float('inf'), float('inf'))
        
        # Check all 4 positions in 2x2 window
        for dy in range(2):
            for dw in range(2):
                val = tl.load(input_ptr + pool_c * 128 * 128 + (src_h + dy) * 128 + (src_w + dw), mask=(src_h + dy) < 128 & (src_w + dw) < 128, other=-float('inf'))
                max_val = tl.maximum(max_val, val)
        
        bypass_interp = max_val  # Bypass interpolation step by direct computation
        bn_input = bypass_interp
    
    # Batch normalization
    total_channels = num_channels_input + num_channels_concat
    norm_idx = out_c
    
    mean = tl.load(running_mean_ptr + norm_idx, mask=norm_idx < total_channels)
    var = tl.load(running_var_ptr + norm_idx, mask=norm_idx < total_channels)
    gamma = tl.load(weight_ptr + norm_idx, mask=norm_idx < total_channels)
    beta = tl.load(bias_ptr + norm_idx, mask=norm_idx < total_channels)
    
    # Batch norm formula: (x - mean) * (gamma / sqrt(var + epsilon)) + beta
    denom = tl.sqrt(var + epsilon)
    scale = gamma / denom
    bias_norm = beta - mean * scale
    
    bn_output = bn_input * scale + bias_norm
    
    # ReLU activation
    relu_output = tl.maximum(bn_output, 0.0)
    
    # Store result
    output_ptr_offset = out_c * 64 * 64 + out_h * 64 + out_w
    tl.store(output_ptr + output_ptr_offset, relu_output, mask=mask)

@torch.fx.wrap
def fused_pool_interp_cat_bn_relu_64(
    input_tensor,           # Result of max_pool (128x128)
    concat_tensor,          # Tensor to concatenate with (64x64)
    running_mean,           # Batch norm running mean
    running_var,            # Batch norm running variance  
    weight,                 # Batch norm weight
    bias,                   # Batch norm bias
):
    batch_size, num_channels_input, height, width = input_tensor.shape
    
    # Extract concatenation tensor shape
    _, num_channels_concat, _, _ = concat_tensor.shape
    
    # Create output tensor: [batch, total_channels, 64, 64]
    total_channels = num_channels_input + num_channels_concat
    output_shape = (batch_size, total_channels, 64, 64)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block sizes for Triton kernel
    BLOCK_SIZE_M = 8   # Height block (64/8 = 8 blocks)
    BLOCK_SIZE_N = 8   # Width block (64/8 = 8 blocks)
    BLOCK_SIZE_K = 64  # Channel block (total_channels/64 blocks)
    
    # Calculate grid dimensions
    grid_h = (64 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_w = (64 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_m = (total_channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Launch kernel
    fused_pool_interp_cat_bn_relu_kernel_64[(
        grid_h, 
        grid_w, 
        grid_m
    )](
        input_tensor,
        concat_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        
        batch_size,
        num_channels_concat,
        num_channels_input,
        128,  # Original height for max_pool indexing
        128,  # Original width for max_pool indexing
        
        # Parameters
        0.001, 0.1,  # epsilon, momentum
        
        # Block sizes
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return output

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern matching: max_pool2d + interpolate(64,64) + cat + batch_norm + relu
    """
    tmp_4 = torch.nn.functional.max_pool2d(in_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, (64, 64), None, 'bilinear', False)
    tmp_6 = torch.cat([in_4, tmp_5], 1)
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    return tmp_8

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

def replacement_func():
    return fused_pool_interp_cat_bn_relu_64