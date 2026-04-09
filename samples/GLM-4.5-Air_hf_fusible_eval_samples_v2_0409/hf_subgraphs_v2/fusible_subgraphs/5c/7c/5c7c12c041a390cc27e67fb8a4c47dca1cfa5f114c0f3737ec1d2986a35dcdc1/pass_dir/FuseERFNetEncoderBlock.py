import torch
import triton
import triton.language as tl
import math

# Pattern matching function - matches the exact computation pattern from model.py
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match ERFNet encoder block pattern: max_pool2d -> interpolate -> cat -> batch_norm -> relu
    Pattern based on the computation from model.py files
    """
    tmp_4 = torch.nn.functional.max_pool2d(in_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, (64, 64), None, 'bilinear', False)
    tmp_6 = torch.cat([in_4, tmp_5], 1)
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    return tmp_8



# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

# Triton kernel for the fused ERFNet encoder block
@triton.jit
def fused_erfnet_encoder_kernel(
    running_mean_ptr,      # Batch norm running mean
    running_var_ptr,       # Batch norm running var  
    weight_ptr,            # Batch norm weight
    bias_ptr,              # Batch norm bias
    input_ptr,             # First input to concatenate
    pooled_ptr,            # Max-pooled intermediate
    out_ptr,               # Output relu
    batch_size,            # Batch size
    channels_input,        # Channels in first input
    channels_pooled,       # Channels in max-pooled input
    height,                # Target height (128)
    width,                 # Target width (128)
    eps: tl.constexpr,     # Batch norm epsilon
    momentum: tl.constexpr, # Batch norm momentum
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    n_elements = batch_size * channels_input * height * width
    
    # Process batch-wise
    batch_offset = pid * BLOCK_SIZE
    offsets = batch_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    if tl.any(mask):
        # Calculate indices
        offset = offsets[0]
        b = offset // (channels_input * height * width)
        c = (offset // (height * width)) % channels_input
        h = (offset // width) % height
        w = offset % width
        
        # Get normalized coordinates for interpolation
        src_h = h * 2  # Map back to original height (256)
        src_w = w * 2  # Map back to original width (256)
        
        # Load input values
        input_val = tl.load(input_ptr + offset, mask=mask, other=0.0)
        
        # Load interpolated pooled value (simplified - bilinear interpolation)
        pooled_idx = b * channels_pooled * height * width + c * height * width + h * width + w
        pooled_val = tl.load(pooled_ptr + pooled_idx, mask=mask, other=0.0)
        
        # Concatenate (channel dimension)
        if c < channels_input:
            cat_val = input_val
        else:
            cat_val = pooled_val
        
        # Batch normalization (simplified for performance)
        mean_val = tl.load(running_mean_ptr + (c % channels_input), mask=None)
        var_val = tl.load(running_var_ptr + (c % channels_input), mask=None)
        weight_val = tl.load(weight_ptr + (c % channels_input), mask=None)
        bias_val = tl.load(bias_ptr + (c % channels_input), mask=None)
        
        # BN computation
        norm_val = (cat_val - mean_val) / tl.sqrt(var_val + eps)
        bn_val = weight_val * norm_val + bias_val
        
        # ReLU activation
        relu_val = tl.maximum(bn_val, 0.0)
        
        # Store result
        tl.store(out_ptr + offset, relu_val, mask=mask)

# Wrapper function that sets up and launches the kernel
@torch.fx.wrap
def fused_erfnet_encoder_wrapper(running_mean, running_var, weight, bias, input_tensor, pooled_output):
    # Get tensor shapes and properties
    batch_size = input_tensor.shape[0]
    channels_input = input_tensor.shape[1] 
    channels_pooled = pooled_output.shape[1]
    height, width = 128, 128
    
    # Determine output shape (same as input after pooling/interpolation)
    output_shape = (batch_size, channels_input + channels_pooled, height, width)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid size
    total_elements = batch_size * (channels_input + channels_pooled) * height * width
    BLOCK_SIZE = 1024
    grid_size = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Launch kernel
    fused_erfnet_encoder_kernel[grid_size](
        running_mean, running_var, weight, bias,
        input_tensor, pooled_output, output,
        batch_size, channels_input, channels_pooled, height, width,
        0.001, 0.1,  # eps, momentum
        BLOCK_SIZE
    )
    
    return output

# Specialized version for different target sizes (64x64)
@triton.jit
def fused_erfnet_encoder_kernel_64(
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    input_ptr, pooled_ptr, out_ptr,
    batch_size, channels_input, channels_pooled,
    height, width,
    eps: tl.constexpr, momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_elements = batch_size * channels_input * height * width
    
    batch_offset = pid * BLOCK_SIZE
    offsets = batch_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    if tl.any(mask):
        offset = offsets[0]
        b = offset // (channels_input * height * width)
        c = (offset // (height * width)) % channels_input
        h = (offset // width) % height
        w = offset % width
        
        # Map back to original coordinates (128x128)
        src_h = h * 2  
        src_w = w * 2
        
        input_val = tl.load(input_ptr + offset, mask=mask, other=0.0)
        pooled_idx = b * channels_pooled * height * width + c * height * width + h * width + w
        pooled_val = tl.load(pooled_ptr + pooled_idx, mask=mask, other=0.0)
        
        if c < channels_input:
            cat_val = input_val
        else:
            cat_val = pooled_val
        
        mean_val = tl.load(running_mean_ptr + (c % channels_input), mask=None)
        var_val = tl.load(running_var_ptr + (c % channels_input), mask=None)
        weight_val = tl.load(weight_ptr + (c % channels_input), mask=None)
        bias_val = tl.load(bias_ptr + (c % channels_input), mask=None)
        
        norm_val = (cat_val - mean_val) / tl.sqrt(var_val + eps)
        bn_val = weight_val * norm_val + bias_val
        relu_val = tl.maximum(bn_val, 0.0)
        
        tl.store(out_ptr + offset, relu_val, mask=mask)

@torch.fx.wrap
def fused_erfnet_encoder_wrapper_64(running_mean, running_var, weight, bias, input_tensor, pooled_output):
    batch_size = input_tensor.shape[0]
    channels_input = input_tensor.shape[1] 
    channels_pooled = pooled_output.shape[1]
    height, width = 64, 64
    
    output_shape = (batch_size, channels_input + channels_pooled, height, width)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    total_elements = batch_size * (channels_input + channels_pooled) * height * width
    BLOCK_SIZE = 1024
    grid_size = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    fused_erfnet_encoder_kernel_64[grid_size](
        running_mean, running_var, weight, bias,
        input_tensor, pooled_output, output,
        batch_size, channels_input, channels_pooled, height, width,
        0.001, 0.1,
        BLOCK_SIZE
    )
    
    return output

# Replacement function (returns function reference, no arguments)
def replacement_func():
    # Return a wrapper that can handle both target sizes
    def multi_target_wrapper(running_mean, running_var, weight, bias, input_tensor, pooled_output):
        # Get target size from pooled_output - we can determine target from its shape
        # The pooled output should be the interpolated version
        
        # Create a dummy computation to determine target size
        # This is a heuristic - we assume if pooled_output > input_tensor in spatial dims, it's 128x128
        if pooled_output.shape[2] > input_tensor.shape[2]:
            return fused_erfnet_encoder_wrapper(running_mean, running_var, weight, bias, input_tensor, pooled_output)
        else:
            return fused_erfnet_encoder_wrapper_64(running_mean, running_var, weight, bias, input_tensor, pooled_output)
    
    return multi_target_wrapper