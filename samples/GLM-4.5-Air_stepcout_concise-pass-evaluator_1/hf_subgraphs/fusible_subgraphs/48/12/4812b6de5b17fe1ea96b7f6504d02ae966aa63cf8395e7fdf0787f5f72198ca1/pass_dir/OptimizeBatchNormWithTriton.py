import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation pattern from model.py
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match batch normalization + slicing pattern"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_5[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    tmp_5 = torch.nn.functional.batch_norm(in_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """Extract arguments needed for the replacement"""
    return (in_0, in_1, in_2, in_3, in_4, in_5)

# Optimized batch normalization kernel using Triton
@triton.jit
def batch_norm_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized batch normalization kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Handle device differences - move parameters to GPU if needed
    if running_mean_ptr is not None:
        running_mean = tl.load(running_mean_ptr + offsets, mask=mask, other=0.0)
    else:
        running_mean = 0.0
        
    if running_var_ptr is not None:
        running_var = tl.load(running_var_ptr + offsets, mask=mask, other=1.0)
    else:
        running_var = 1.0
        
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    else:
        weight = 1.0
        
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    else:
        bias = 0.0
    
    # Apply batch normalization formula: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (input_data - running_mean) * tl.rsqrt(running_var + eps) * weight + bias
    
    # Store result
    tl.store(output_ptr + offsets, normalized, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(input_tensor, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=0.001):
    """Optimized batch normalization using Triton kernel"""
    # Ensure parameters are on the same device as input
    device = input_tensor.device
    if running_mean.device != device:
        running_mean = running_mean.to(device)
    if running_var.device != device:
        running_var = running_var.to(device)
    if weight.device != device:
        weight = weight.to(device)
    if bias.device != device:
        bias = bias.to(device)
    
    # Get input tensor shape
    shape = input_tensor.shape
    n_elements = input_tensor.numel()
    
    # Set block size and grid size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch Triton kernel
    batch_norm_kernel[(num_programs,)](
        output_ptr=output,
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@torch.fx.wrap
def optimized_forward(in_0, in_1, in_2, in_3, in_4, in_5):
    """Optimized forward function that combines batch norm and slicing"""
    # Perform slicing operation (this is already efficient in PyTorch)
    tmp_4 = in_5[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    
    # Perform optimized batch normalization
    tmp_5 = optimized_batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    
    return (tmp_5, tmp_4)

# Replacement function (returns function reference, not a call)
def replacement_func():
    return optimized_forward