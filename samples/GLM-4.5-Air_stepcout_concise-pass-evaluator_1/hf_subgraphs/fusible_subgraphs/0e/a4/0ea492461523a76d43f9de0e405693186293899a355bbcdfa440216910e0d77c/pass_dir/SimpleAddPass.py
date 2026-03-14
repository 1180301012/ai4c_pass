import torch
import triton
import triton.language as tl

def pattern(input, running_mean, running_var, weight, bias):
    output = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return output

def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias)

@triton.jit
def simple_add_kernel(input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, 
                      output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, eps=1e-05):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    running_mean = tl.load(running_mean_ptr)  # Scalar parameter
    running_var = tl.load(running_var_ptr)    # Scalar parameter
    weight = tl.load(weight_ptr)              # Scalar parameter
    bias = tl.load(bias_ptr)                  # Scalar parameter
    
    # Perform batch normalization: (x * weight + bias - running_mean) / sqrt(running_var + eps)
    normalized = (x * weight + bias - running_mean) / tl.sqrt(running_var + eps)
    
    # Store result
    tl.store(output_ptr + offsets, normalized, mask=mask)

@torch.fx.wrap
def fused_batch_norm(input, running_mean, running_var, weight, bias):
    N = input.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Launch kernel
    simple_add_kernel[(num_programs,)](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=1e-05,
    )
    
    return output

def replacement_func():
    return fused_batch_norm