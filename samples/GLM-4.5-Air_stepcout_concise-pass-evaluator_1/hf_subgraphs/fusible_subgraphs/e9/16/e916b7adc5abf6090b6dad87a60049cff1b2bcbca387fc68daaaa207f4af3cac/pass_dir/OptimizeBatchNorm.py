import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
    # Match the batch normalization and the intermediate result
    tmp_10 = in_7  # This is the result from previous operations
    tmp_3 = in_3
    tmp_4 = in_4
    tmp_6 = in_6
    tmp_5 = in_5
    
    # Match exact batch_norm call from model.py
    # torch.nn.functional.batch_norm(tmp_10, tmp_3, tmp_4, tmp_6, tmp_5, False, 0.1, 1e-05)
    tmp_11 = torch.nn.functional.batch_norm(tmp_10, tmp_3, tmp_4, tmp_6, tmp_5, False, 0.1, 1e-05)
    
    # Return what the original returns: (batch_norm_result, intermediate_result)
    return tmp_11, tmp_10

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
    return (in_7, in_3, in_4, in_6, in_5)

@triton.jit
def batch_norm_kernel(
    output_ptr,
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    
    # Shape information
    num_features: tl.constexpr,
    batch_size: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * height * width)
    
    # Load input
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute spatial position and feature index
    spatial_idx = offsets % (height * width)
    feature_idx = offsets // (height * width)
    
    # Load batch norm parameters
    rmean = tl.load(running_mean_ptr + feature_idx, other=0.0)
    rvar = tl.load(running_var_ptr + feature_idx, other=1.0)
    
    # Load weight and bias
    weight_val = tl.load(weight_ptr + feature_idx, other=1.0)
    bias_val = tl.load(bias_ptr + feature_idx, other=0.0)
    
    # Batch norm computation: (x - mean) / sqrt(var + eps) * weight + bias
    # For inference, we use running mean and var
    denom = tl.sqrt(rvar + eps)
    normalized = (input_val - rmean) / denom
    output = normalized * weight_val + bias_val
    
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(input_tensor, running_mean, running_var, weight, bias):
    # Get tensor shape information
    batch_size, num_features, height, width = input_tensor.shape
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Block size for GPU optimization
    BLOCK_SIZE = 1024
    total_elements = batch_size * height * width
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with batch norm parameters
    batch_norm_kernel[(num_programs,)](
        output,
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        
        num_features,
        batch_size,
        height,
        width,
        0.1,    # momentum (not used in inference)
        1e-05,  # epsilon
        
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_batch_norm