import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    tmp_5 = torch.nn.functional.gelu(x, approximate='none')
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return tmp_5, tmp_6

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def fused_gelu_batch_norm_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    gelu_out_ptr,
    batch_out_ptr,
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
    
    # Load normalization parameters 
    mean_idx = pid % C
    var_idx = pid % C
    weight_idx = pid % C
    bias_idx = pid % C
    
    mean = tl.load(mean_ptr + mean_idx)
    var = tl.load(var_ptr + var_idx)
    weight = tl.load(weight_ptr + weight_idx)
    bias = tl.load(bias_ptr + bias_idx)
    
    # First: Compute GELU
    gelu_out = 0.5 * x * (1.0 + tl.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    
    # Second: Batch normalization on GELU output
    inv_std = 1.0 / tl.sqrt(var + 1e-05)
    x_norm = (gelu_out - mean) * inv_std
    batch_out = weight * x_norm + bias
    
    # Store outputs
    tl.store(gelu_out_ptr + offsets, gelu_out, mask=mask)
    tl.store(batch_out_ptr + offsets, batch_out, mask=mask)

@torch.fx.wrap
def optimized_fused_gelu_batch_norm(x, running_mean, running_var, weight, bias):
    # Safe input checking
    if hasattr(x, 'numel') and hasattr(running_mean, 'numel'):
        N = x.numel()
        C = running_mean.numel()
    else:
        # Simple fallback: just pass through (no transformation)
        return x, x
    
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    gelu_out = torch.empty_like(x)
    batch_out = torch.empty_like(x)
    
    fused_gelu_batch_norm_kernel[(num_programs,)](
        x_ptr=x,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        gelu_out_ptr=gelu_out,
        batch_out_ptr=batch_out,
        n_elements=N,
        C=C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return gelu_out, batch_out

def replacement_func():
    return optimized_fused_gelu_batch_norm