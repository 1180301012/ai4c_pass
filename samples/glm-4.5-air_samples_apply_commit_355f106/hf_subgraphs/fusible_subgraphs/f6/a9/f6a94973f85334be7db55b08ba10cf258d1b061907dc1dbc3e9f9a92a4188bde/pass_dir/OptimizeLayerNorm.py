import torch
import triton
import triton.language as tl
import math

def pattern(x, weight, bias):
    # torch.nn.functional.layer_norm(tmp_14, (768,), tmp_3, tmp_2, 1e-05)
    return torch.nn.functional.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=1e-05)

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_elements, hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program processes a contiguous block of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate mean for normalization
    x_mean = tl.sum(x, axis=0) / hidden_size
    x_mean2 = tl.sum(x * x, axis=0) / hidden_size
    
    # Calculate variance and standard deviation
    x_var = x_mean2 - x_mean * x_mean
    x_std = tl.sqrt(x_var + eps)
    
    # Normalize
    x_normalized = (x - tl.full((hidden_size,), x_mean)) / tl.full((hidden_size,), x_std)
    
    # Apply weights and bias
    weight = tl.load(weight_ptr + offsets % hidden_size, mask=offsets % hidden_size < hidden_size, other=1.0)
    bias = tl.load(bias_ptr + offsets % hidden_size, mask=offsets % hidden_size < hidden_size, other=0.0)
    
    out = x_normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias):
    batch_size, seq_len, hidden_size = x.shape
    n_elements = batch_size * seq_len * hidden_size
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    layer_norm_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        hidden_size=hidden_size,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return optimized_layer_norm