import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias, prelu_w, training, momentum, eps):
    # batch_norm
    batch_norm_out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
    # prelu
    prelu_out = torch.prelu(batch_norm_out, prelu_w)
    return batch_norm_out, prelu_out

def replacement_args(x, running_mean, running_var, weight, bias, prelu_w, training, momentum, eps):
    return (x, running_mean, running_var, weight, bias, prelu_w, training, momentum, eps)

@triton.jit
def fused_batch_norm_prelu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    prelu_weight_ptr,
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
    
    # Load batch norm parameters
    weight = tl.load(weight_ptr + offsets % tl.numel(weight_ptr), mask=offsets % tl.numel(weight_ptr) < tl.numel(weight_ptr))
    bias = tl.load(bias_ptr + offsets % tl.numel(bias_ptr), mask=offsets % tl.numel(bias_ptr) < tl.numel(bias_ptr))
    running_mean = tl.load(running_mean_ptr + offsets % tl.numel(running_mean_ptr), mask=offsets % tl.numel(running_mean_ptr) < tl.numel(running_mean_ptr))
    running_var = tl.load(running_var_ptr + offsets % tl.numel(running_var_ptr), mask=offsets % tl.numel(running_var_ptr) < tl.numel(running_var_ptr))
    prelu_weight_val = tl.load(prelu_weight_ptr)
    
    # Normalize variance
    std = tl.sqrt(running_var + 1e-5)
    
    # Apply batch normalization
    x_normalized = (x - running_mean) / std
    batch_norm_out = x_normalized * weight + bias
    
    # Apply PReLU
    prelu_out = tl.where(batch_norm_out > 0, batch_norm_out, batch_norm_out * prelu_weight_val)
    
    # Store output
    tl.store(out_ptr + offsets, prelu_out, mask=mask)

@torch.fx.wrap
def fused_batch_norm_prelu(x, weight, bias, running_mean, running_var, prelu_weight):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Determine output shape based on input
    output_shape = x.shape
    
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    fused_batch_norm_prelu_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        prelu_weight_ptr=prelu_weight,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_batch_norm_prelu