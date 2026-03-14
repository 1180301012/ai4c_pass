import torch
import triton
import triton.language as tl
from torch import device


# Pattern matching function
def pattern(a, b, weight, bias):
    """
    Match the pattern:
    - add two tensors
    - layer norm
    """
    added = a + b
    normalized = torch.nn.functional.layer_norm(added, (1024,), weight, bias, 1e-05)
    return normalized


def replacement_args(a, b, weight, bias):
    return (a, b, weight, bias)


@triton.jit
def fused_add_layernorm_kernel(
    a_ptr,
    b_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    output_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Fused kernel that:
    1. Adds two tensors
    2. Applies layer normalization
    Each program processes one row (sequence element)
    """
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Calculate row offset
    row_offset = row_idx * N
    
    # Load data
    offsets = tl.arange(0, 1024)
    
    a_val = tl.load(a_ptr + row_offset + offsets)
    b_val = tl.load(b_ptr + row_offset + offsets)
    ln_weight = tl.load(ln_weight_ptr + offsets)
    ln_bias = tl.load(ln_bias_ptr + offsets)
    
    # Add
    x = a_val + b_val
    
    # Layer norm: compute mean
    mean = tl.sum(x, axis=0) * (1.0 / N)
    
    # Variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) * (1.0 / N)
    
    # Normalize and apply affine transformation in one step
    rstd = 1.0 / tl.sqrt(var + eps)
    output = x_centered * rstd * ln_weight + ln_bias
    
    # Store output
    tl.store(output_ptr + row_offset + offsets, output)


@torch.fx.wrap
def fused_add_layernorm(a, b, weight, bias):
    """
    Fused implementation combining:
    - Addition
    - Layer normalization (dim=1024, eps=1e-05)
    """
    # Flatten to 2D for processing
    orig_shape = a.shape
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    
    batch_seq_size, hidden_dim = a_flat.shape
    eps = 1e-05
    
    # Output tensor
    output = torch.empty_like(a_flat)
    
    # Launch one kernel per sequence element in parallel
    grid = (batch_seq_size,)
    fused_add_layernorm_kernel[grid](
        a_flat,
        b_flat,
        weight,
        bias,
        output,
        N=hidden_dim,
        eps=eps,
        num_warps=4,
    )
    
    return output.reshape(orig_shape)


def replacement_func():
    return fused_add_layernorm