import torch
import triton
import triton.language as tl
import math

def pattern(x, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    return (x, normalized_shape[0], weight, bias, eps)

@triton.jit
def layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    channels,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel of the input
    pid = tl.program_id(0)
    if pid >= channels:
        return
    
    # Process a block of elements for this channel
    offsets = pid + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs for this channel
    x = tl.load(x_ptr + offsets * channels + pid, mask=mask, other=0.0)
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + pid)
    else:
        weight = 1.0
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid)
    else:
        bias = 0.0
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n_elements
    x_centered = x - mean
    
    # Compute variance
    x2 = x_centered * x_centered
    var = tl.sum(x2, axis=0) / n_elements
    
    # Normalize and apply weight/bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x_centered * inv_std) * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets * channels + pid, out, mask=mask)

@torch.fx.wrap
def optimized_layernorm(x, weight, bias, eps):
    # Get input shape
    if x.dim() == 3:
        batch, seq_len, channels = x.shape
        total_elements = batch * seq_len
    else:
        raise ValueError("Only 3D inputs supported")
    
    # Set up launch configuration
    BLOCK_SIZE = 1024
    grid_size = (channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    layernorm_kernel[grid_size](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=total_elements,
        channels=channels,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_layernorm