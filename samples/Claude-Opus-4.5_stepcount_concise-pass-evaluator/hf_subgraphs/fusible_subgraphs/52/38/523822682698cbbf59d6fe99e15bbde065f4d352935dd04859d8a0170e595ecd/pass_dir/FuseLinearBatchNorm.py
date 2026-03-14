import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias):
    """
    Match batch normalization inference pattern.
    """
    out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return out


def replacement_args(x, running_mean, running_var, weight, bias):
    """Extract arguments needed for the replacement function."""
    return (x, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def batchnorm_inference_kernel(
    x_ptr,
    mean_ptr,
    var_ptr, 
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    features,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized batch normalization kernel for inference mode.
    Computes: out = (x - mean) / sqrt(var + eps) * weight + bias
    """
    pid = tl.program_id(0)
    
    # Process BLOCK_SIZE elements per program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate feature indices for parameter lookup
    feature_idx = offsets % features
    
    # Load parameters (same across batch dimension)
    mean = tl.load(mean_ptr + feature_idx, mask=mask)
    var = tl.load(var_ptr + feature_idx, mask=mask)
    weight = tl.load(weight_ptr + feature_idx, mask=mask)
    bias = tl.load(bias_ptr + feature_idx, mask=mask)
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute batch norm: (x - mean) * rsqrt(var + eps) * weight + bias
    inv_std = tl.rsqrt(var + eps)
    normalized = (x - mean) * inv_std
    out = normalized * weight + bias
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def batchnorm_optimized(x, running_mean, running_var, weight, bias):
    """
    Optimized batch normalization using Triton kernel.
    """
    batch_size = x.shape[0]
    features = x.shape[1]
    n_elements = batch_size * features
    
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    batchnorm_inference_kernel[grid](
        x,              # x_ptr
        running_mean,   # mean_ptr
        running_var,    # var_ptr
        weight,         # weight_ptr
        bias,           # bias_ptr
        out,            # out_ptr
        n_elements,
        features,
        eps=1e-05,
    )
    
    return out


def replacement_func():
    return batchnorm_optimized