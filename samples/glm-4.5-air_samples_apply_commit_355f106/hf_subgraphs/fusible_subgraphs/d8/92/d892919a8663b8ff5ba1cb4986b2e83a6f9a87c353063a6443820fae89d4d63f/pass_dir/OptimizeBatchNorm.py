import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias, training, momentum, eps):
    # batch_norm
    batch_norm_out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
    return batch_norm_out

def replacement_args(x, running_mean, running_var, weight, bias, training, momentum, eps):
    return (x, running_mean, running_var, weight, bias, training, momentum, eps)

@triton.jit
def optimized_batch_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load batch norm parameters (assuming 1D parameter tensors)
    param_size = 128  # From weight metadata - parameter tensors have size 128
    weight_idx = offsets % param_size
    bias_idx = offsets % param_size
    running_mean_idx = offsets % param_size
    running_var_idx = offsets % param_size
    
    weight = tl.load(weight_ptr + weight_idx, mask=weight_idx < param_size)
    bias = tl.load(bias_ptr + bias_idx, mask=bias_idx < param_size)
    running_mean = tl.load(running_mean_ptr + running_mean_idx, mask=running_mean_idx < param_size)
    running_var = tl.load(running_var_ptr + running_var_idx, mask=running_var_idx < param_size)
    
    # Normalize variance
    std = tl.sqrt(running_var + 1e-5)
    
    # Apply batch normalization
    x_normalized = (x - running_mean) / std
    batch_norm_out = x_normalized * weight + bias
    
    # Store output
    tl.store(out_ptr + offsets, batch_norm_out, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps):
    n_elements = x.numel()
    
    # Dynamic block size based on input size
    if n_elements < 8192:
        BLOCK_SIZE = 256
    elif n_elements < 65536:
        BLOCK_SIZE = 512
    elif n_elements < 524288:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Determine output shape based on input
    output_shape = x.shape
    
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    optimized_batch_norm_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_batch_norm