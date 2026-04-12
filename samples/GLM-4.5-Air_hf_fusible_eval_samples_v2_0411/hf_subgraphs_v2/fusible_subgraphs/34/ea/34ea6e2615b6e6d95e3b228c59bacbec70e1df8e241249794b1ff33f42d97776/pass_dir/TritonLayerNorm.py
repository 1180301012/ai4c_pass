import torch
import triton
import triton.language as tl

def pattern(tmp_7, in_1, in_0):
    """Match layer normalization pattern"""
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_1, in_0, 1e-05)
    return tmp_8

def replacement_args(tmp_7, in_1, in_0):
    return (tmp_7, in_1, in_0, 768)

@triton.jit
def layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance layer normalization kernel"""
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load data for this row
    x = tl.load(x_ptr + row * n_cols + col_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    # Compute mean
    x_mean = tl.sum(x, axis=0) / n_cols
    
    # Compute variance
    x_var = tl.sum((x - x_mean) * (x - x_mean), axis=0) / n_cols
    x_var = tl.where(x_var < 0.0, 0.0, x_var)  # Clip negative variance to 0
    
    # Normalize
    x_norm = (x - x_mean) * tl.math.rsqrt(x_var + eps)
    
    # Apply weight and bias
    out = x_norm * weight + bias
    
    # Store result
    tl.store(out_ptr + row * n_cols + col_offsets, out, mask=mask)

@torch.fx.wrap
def triton_layernorm(x, weight, bias, normalized_shape, eps=1e-05):
    """High-performance layer normalization using Triton"""
    # Determine tensor dimensions
    if len(x.shape) == 3:
        n_rows, n_cols = x.shape[0], x.shape[2]  # After flatten and transpose: [seq_len, batch=1, features]
    elif len(x.shape) == 2:
        n_rows, n_cols = x.shape[0], x.shape[1]
    else:
        raise ValueError(f"Unsupported tensor shape for layernorm: {x.shape}")
    
    BLOCK_SIZE = 1024
    num_programs = (n_rows * n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    if weight.dtype == torch.bfloat16:
        eps = tl.float32(eps)
    
    layernorm_kernel[num_programs](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_cols=n_cols,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_layernorm