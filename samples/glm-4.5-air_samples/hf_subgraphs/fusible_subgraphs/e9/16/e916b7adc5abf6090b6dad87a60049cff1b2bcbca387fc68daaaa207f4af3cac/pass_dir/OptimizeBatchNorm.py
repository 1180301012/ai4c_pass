import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def batch_norm_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    batch, cin, h, w, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    total_elements = batch * cin * h * w
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load weight and bias (these are typically constants from metadata)
    weight_val = tl.load(weight_ptr, other=1.0)
    bias_val = tl.load(bias_ptr, other=0.0)
    mean_val = tl.load(mean_ptr, other=0.0)
    var_val = tl.load(var_ptr, other=1.0)
    
    for offset in offsets[mask]:
        # Calculate 4D coordinates
        batch_idx = offset // (cin * h * w)
        remaining = offset % (cin * h * w)
        c_idx = remaining // (h * w)
        spatial_idx = remaining % (h * w)
        h_idx = spatial_idx // w
        w_idx = spatial_idx % w
        
        # Load input value
        x_val = tl.load(x_ptr + offset, other=0.0)
        
        # Batch norm: (x - mean) / sqrt(var + eps) * weight + bias
        normalized = (x_val - mean_val) / tl.sqrt(var_val + eps)
        out_val = normalized * weight_val + bias_val
        
        # Store result
        tl.store(out_ptr + offset, out_val, other=0.0)

@torch.fx.wrap
def optimized_batch_norm(x, running_mean, running_var, weight, bias):
    batch, cin, h, w = x.shape
    eps = 1e-05
    
    total_elements = batch * cin * h * w
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    batch_norm_kernel[(num_programs,)](
        x_ptr=x, mean_ptr=running_mean, var_ptr=running_var,
        weight_ptr=weight, bias_ptr=bias, out_ptr=out,
        batch=batch, cin=cin, h=h, w=w, eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return optimized_batch_norm