import torch
import triton
import triton.language as tl

# Pattern matching for layer_norm with fixed normalized shape (256)
def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (256,), weight, bias, 1e-05)

# Extract arguments needed for replacement

def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Triton kernel for optimized layer_norm on 256-element vectors
@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_hidden: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Process one vector (256 elements) per block
    idx = tl.program_id(0)
    x_ptr = x_ptr + idx * n_hidden
    
    # Load the entire vector
    x = tl.load(x_ptr + tl.arange(0, n_hidden), 
                mask=tl.arange(0, n_hidden) < n_hidden)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n_hidden
    
    # Compute variance
    var = tl.sum((x - mean) * (x - mean), axis=0) / n_hidden
    
    # Normalize and apply scale/bias
    x = (x - mean) / tl.sqrt(var + eps)
    weight = tl.load(weight_ptr + tl.arange(0, n_hidden))
    bias = tl.load(bias_ptr + tl.arange(0, n_hidden))
    x = x * weight + bias
    
    # Store result
    offsets = idx * n_hidden + tl.arange(0, n_hidden)
    tl.store(out_ptr + offsets, x)

# Kernel wrapper with correct tensor reshaping
@torch.fx.wrap
def layer_norm_wrapper(x, weight, bias):
    # Flatten batch/seq dimensions
    batch = x.shape[0]
    seq = x.shape[1]
    batch_seq = batch * seq
    out = torch.empty_like(x)

    # Launch kernel
    layer_norm_kernel[(batch_seq,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_hidden=256,
        eps=1e-5,
        BLOCK_SIZE=256
    )
    
    return out

# Replacement function (no arguments)
def replacement_func():
    return layer_norm_wrapper