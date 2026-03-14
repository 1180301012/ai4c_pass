import torch
import triton
import triton.language as tl

def pattern(in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    """Pattern matches batch normalization operation:
    tmp_7 = torch.nn.functional.batch_norm(in_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    """
    tmp_7 = torch.nn.functional.batch_norm(in_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    return tmp_7

def replacement_args(in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    return (in_7, tmp_0, tmp_1, tmp_3, tmp_2)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    num_features,
    batch_size,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one feature dimension
    pid = tl.program_id(0)
    
    if pid >= num_features:
        return
    
    # Load parameters for this feature
    mean = tl.load(running_mean_ptr + pid)
    var = tl.load(running_var_ptr + pid)
    weight = tl.load(weight_ptr + pid)
    bias = tl.load(bias_ptr + pid)
    
    # Precompute inverse standard deviation
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Process each element in the batch
    for i in range(0, batch_size):
        offset = pid + i * num_features
        input_val = tl.load(input_ptr + offset)
        
        # Apply batch norm: weight * (x - mean) / sqrt(var + eps) + bias
        output_val = weight * (input_val - mean) * inv_std + bias
        
        tl.store(output_ptr + offset, output_val)

@triton.jit
def batch_norm_kernel_simple(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    num_features,
    batch_size,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one feature dimension
    pid = tl.program_id(0)
    
    if pid >= num_features:
        return
    
    # Load parameters for this feature
    mean = tl.load(running_mean_ptr + pid)
    var = tl.load(running_var_ptr + pid)
    weight_value = tl.load(weight_ptr + pid)
    bias_value = tl.load(bias_ptr + pid)
    
    # Precompute inverse standard deviation
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Process all elements in the batch for this feature
    for i in range(batch_size):
        offset = pid + i * num_features
        input_val = tl.load(input_ptr + offset)
        
        # Apply batch norm: weight * (x - mean) / sqrt(var + eps) + bias
        output_val = weight_value * (input_val - mean) * inv_std + bias_value
        
        tl.store(output_ptr + offset, output_val)

@torch.fx.wrap
def optimized_batch_norm(input, running_mean, running_var, weight, bias):
    # Get tensor shapes
    num_features = running_mean.shape[0]
    batch_size = input.shape[0] if len(input.shape) > 1 else 1
    
    # Use simple approach: one program per feature
    BLOCK_SIZE = 1024
    num_programs = (num_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Apply batch norm
    eps = 1e-05
    batch_norm_kernel_simple[(num_programs,)](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        num_features=num_features,
        batch_size=batch_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_batch_norm