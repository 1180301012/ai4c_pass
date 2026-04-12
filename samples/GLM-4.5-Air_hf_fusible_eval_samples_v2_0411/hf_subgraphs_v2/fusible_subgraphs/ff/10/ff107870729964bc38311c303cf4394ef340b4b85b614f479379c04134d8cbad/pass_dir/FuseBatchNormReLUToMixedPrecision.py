import torch
import triton
import triton.language as tl

@triton.jit
def fused_batchnorm_relu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load batch norm parameters (broadcast across spatial dims)
    mean = tl.load(running_mean_ptr + (offsets // (n_elements // 256)), mask=mask, other=0.0)
    var = tl.load(running_var_ptr + (offsets // (n_elements // 256)), mask=mask, other=0.0)
    weight = tl.load(weight_ptr + (offsets // (n_elements // 256)), mask=mask, other=1.0)
    bias = tl.load(bias_ptr + (offsets // (n_elements // 256)), mask=mask, other=0.0)
    
    # Batch norm computation
    var = tl.maximum(var, eps)
    x_norm = (x - mean) / tl.sqrt(var)
    out = weight * x_norm + bias
    
    # ReLU activation
    out = tl.maximum(out, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_batchnorm_relu(input, running_mean, running_var, weight, bias, eps=1e-05):
    """Fused batch norm + ReLU with optimized precision handling"""
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor in mixed precision
    output = torch.empty_like(input)
    
    # Launch Triton kernel
    fused_batchnorm_relu_kernel[(num_programs,)](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(x, running_mean, running_var, weight, bias, training, momentum, eps):
    """Match batch_norm -> ReLU pattern"""
    out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
    relu_out = torch.nn.functional.relu(out, inplace=False)
    return relu_out

def replacement_args(x, running_mean, running_var, weight, bias, training, momentum, eps):
    return (x, running_mean, running_var, weight, bias, training, momentum, eps)

def replacement_func():
    return fused_batchnorm_relu