import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation pattern from model.py
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match ERFNet encoder block pattern: max_pool2d -> interpolate -> cat -> batch_norm -> relu
    Handles both direct and temporary variable usage for batch_norm parameters
    """
    # Handle temporary variable creation (case 2)
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = torch.nn.functional.max_pool2d(in_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, (128, 128), None, 'bilinear', False)
    tmp_4 = None
    tmp_6 = torch.cat([in_4, tmp_5], 1)
    tmp_5 = None
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 0.001)
    tmp_6 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    tmp_7 = None
    return tmp_8

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

# Simple Triton kernel
@triton.jit
def simple_erfnet_kernel(
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    input_ptr, pooled_ptr, out_ptr,
    batch_size, channels_input, channels_pooled, height, width,
    eps: tl.constexpr,
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
        
        # Load inputs
        input_val = tl.load(input_ptr + offset, mask=mask, other=0.0)
        pooled_idx = b * channels_pooled * height * width + c * height * width + h * width + w
        pooled_val = tl.load(pooled_ptr + pooled_idx, mask=mask, other=0.0)
        
        # Simple concatenation logic
        if c < channels_input:
            cat_val = input_val
        else:
            cat_val = pooled_val
        
        # Batch normalization
        mean_val = tl.load(running_mean_ptr + (c % channels_input), mask=None)
        var_val = tl.load(running_var_ptr + (c % channels_input), mask=None)
        weight_val = tl.load(weight_ptr + (c % channels_input), mask=None)
        bias_val = tl.load(bias_ptr + (c % channels_input), mask=None)
        
        norm_val = (cat_val - mean_val) / tl.sqrt(var_val + eps)
        bn_val = weight_val * norm_val + bias_val
        relu_val = tl.maximum(bn_val, 0.0)
        
        tl.store(out_ptr + offset, relu_val, mask=mask)

# Wrapper function
@torch.fx.wrap  
def simple_erfnet_wrapper(running_mean, running_var, weight, bias, input_tensor, pooled_output):
    batch_size = input_tensor.shape[0]
    channels_input = input_tensor.shape[1] 
    channels_pooled = pooled_output.shape[1]
    height, width = 64, 64
    
    output_shape = (batch_size, channels_input + channels_pooled, height, width)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    total_elements = batch_size * (channels_input + channels_pooled) * height * width
    BLOCK_SIZE = 1024
    grid_size = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    simple_erfnet_kernel[grid_size](
        running_mean, running_var, weight, bias,
        input_tensor, pooled_output, output,
        batch_size, channels_input, channels_pooled, height, width,
        0.001,
        BLOCK_SIZE
    )
    
    return output

# Replacement function
def replacement_func():
    return simple_erfnet_wrapper