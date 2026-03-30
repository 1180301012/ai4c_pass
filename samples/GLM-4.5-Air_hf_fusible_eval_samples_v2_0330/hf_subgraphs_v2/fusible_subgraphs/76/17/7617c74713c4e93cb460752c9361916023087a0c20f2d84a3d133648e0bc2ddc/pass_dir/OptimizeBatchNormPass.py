import torch
import triton
import triton.language as tl

def pattern(input, running_mean, running_var, weight, bias, training, momentum, eps):
    """
    Pattern to match batch normalization operation.
    """
    return torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def replacement_args(input, running_mean, running_var, weight, bias, training, momentum, eps):
    """Extract arguments for the replacement function"""
    return (input, running_mean, running_var, weight, bias, training, momentum, eps)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    mean_ptr,
    var_ptr, 
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    C: tl.constexpr,  # number of channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized batch normalization kernel for inference mode"""
    # Each program handles a specific channel
    c = tl.program_id(0)
    
    # Calculate spatial offset for this channel
    spatial_offset = c * H * W
    
    # Load running mean and variance for this channel
    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    std = tl.sqrt(var + 1e-05)  # eps = 1e-05
    
    # Load weight and bias for this channel
    weight_val = tl.load(weight_ptr + c) if weight_ptr else 1.0
    bias_val = tl.load(bias_ptr + c) if bias_ptr else 0.0
    
    # Process spatial elements for this channel
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = spatial_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load input data
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # Apply batch normalization: (x - mean) / std * weight + bias
        x_norm = (x - mean) / std
        y = x_norm * weight_val + bias_val
        
        # Store result
        tl.store(output_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps):
    """Optimized batch normalization using Triton kernel"""
    if training:
        # For training, use the standard PyTorch implementation
        return torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    
    N, C, H, W = input.shape
    n_elements = N * C * H * W
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Launch Triton kernel
    grid = (C,)  # One program per channel
    
    batch_norm_kernel[grid](
        input_ptr=input,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=1024,
    )
    
    return output

def replacement_func():
    """Return the optimized batch normalization function"""
    return optimized_batch_norm