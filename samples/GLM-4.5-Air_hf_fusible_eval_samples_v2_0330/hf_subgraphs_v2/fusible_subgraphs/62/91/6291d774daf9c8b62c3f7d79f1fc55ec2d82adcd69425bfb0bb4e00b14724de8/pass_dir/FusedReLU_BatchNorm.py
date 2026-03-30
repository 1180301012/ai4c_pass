import torch
import triton
import triton.language as tl

@triton.jit
def fused_relu_batchnorm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + BatchNorm kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU
    x_relu = tl.maximum(x, 0.0)
    
    # Load batch norm parameters
    running_mean = tl.load(running_mean_ptr + offsets, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + offsets, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # BatchNorm computation: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (x_relu - running_mean) / tl.sqrt(running_var + eps)
    output = normalized * weight + bias
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def fused_relu_batchnorm(input_tensor, running_mean, running_var, weight, bias):
    """Fused ReLU + BatchNorm operation"""
    # Get input shape
    input_shape = input_tensor.shape
    n_elements = input_tensor.numel()
    
    # Block size for processing
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    fused_relu_batchnorm_kernel[(num_programs,)](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(x, running_mean, running_var, weight, bias):
    """Pattern matches ReLU followed by BatchNorm"""
    # First apply ReLU
    relu_out = torch.nn.functional.relu(x, inplace=False)
    
    # Then apply BatchNorm - using exact same signature as original (positional args)
    batchnorm_out = torch.nn.functional.batch_norm(
        relu_out, running_mean, running_var, weight, bias, False, 0.1, 1e-05
    )
    
    return batchnorm_out

def replacement_args(x, running_mean, running_var, weight, bias):
    """Extract arguments for the fused operation"""
    return (x, running_mean, running_var, weight, bias)

def replacement_func():
    """Return the fused kernel function"""
    return fused_relu_batchnorm