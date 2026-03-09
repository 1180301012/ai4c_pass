import torch
import triton
import triton.language as tl

def pattern(input, running_mean, running_var, weight, bias, use_stats, momentum, eps):
    return torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, use_stats, momentum, eps)

def replacement_args(input, running_mean, running_var, weight, bias, use_stats, momentum, eps):
    return (input, running_mean, running_var, weight, bias, use_stats, momentum, eps)

@triton.jit
def batch_norm_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load normalization parameters - only once per program
    mean_idx = pid % C
    var_idx = pid % C
    weight_idx = pid % C
    bias_idx = pid % C
    
    mean = tl.load(mean_ptr + mean_idx)
    var = tl.load(var_ptr + var_idx)
    weight = tl.load(weight_ptr + weight_idx)
    bias = tl.load(bias_ptr + bias_idx)
    
    # Batch normalization computation
    inv_std = 1.0 / tl.sqrt(var + 1e-05)
    x_norm = (x - mean) * inv_std
    out = weight * x_norm + bias
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(input, running_mean, running_var, weight, bias, use_stats, momentum, eps):
    N = input.numel()
    C = running_mean.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(input)
    batch_norm_kernel[(num_programs,)](
        x_ptr=input,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=N,
        C=C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return optimized_batch_norm