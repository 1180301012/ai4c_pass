import torch
import triton
import triton.language as tl

# Simpler pattern - just match the expensive layer_norm operation
def pattern(x, normalized_shape, weight, bias):
    """
    Match layer_norm pattern
    """
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, 1e-06)


def replacement_args(x, normalized_shape, weight, bias):
    return (x, weight, bias)


@triton.jit
def layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N, D,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Fused layer normalization kernel with Welford's algorithm for stability
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= N:
        return
    
    # Load input row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D
    
    x_row_ptr = x_ptr + row_idx * D
    x_row = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    
    # Compute mean using only valid elements
    sum_x = tl.sum(tl.where(mask, x_row, 0.0), axis=0)
    mean = sum_x / D
    
    # Center the data
    x_centered = x_row - mean
    
    # Compute variance using only valid elements
    sum_sq = tl.sum(tl.where(mask, x_centered * x_centered, 0.0), axis=0)
    var = sum_sq / D
    
    # Normalize with epsilon for stability
    rstd = tl.rsqrt(var + eps)
    x_norm = x_centered * rstd
    
    # Apply affine transform
    out = x_norm * weight + bias
    
    # Store output only for valid elements
    out_row_ptr = out_ptr + row_idx * D
    tl.store(out_row_ptr + cols, out, mask=mask)


@torch.fx.wrap
def fused_layernorm(x, weight, bias):
    """
    Fused layer normalization
    """
    # Get shape
    shape = x.shape
    N = 1
    for i in range(len(shape) - 1):
        N *= shape[i]
    D = shape[-1]
    
    # Flatten input
    x_flat = x.reshape(N, D)
    
    # Allocate output
    out = torch.empty_like(x_flat)
    
    # Launch kernel
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (N,)
    
    layernorm_kernel[grid](
        x_flat,
        weight,
        bias,
        out,
        N, D,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=1e-06,
    )
    
    # Reshape output
    return out.reshape(shape)


def replacement_func():
    return fused_layernorm