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


@triton.jit
def bn_kernel(
    x_ptr,
    mean_ptr,
    var_ptr, 
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    features,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized batch normalization kernel for inference.
    Computes: out = (x - mean) * (weight / sqrt(var + eps)) + bias
            = x * scale + shift
    where scale = weight * rsqrt(var + eps), shift = bias - mean * scale
    """
    pid = tl.program_id(0)
    eps = 1e-05
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    feat_idx = offsets % features
    
    # Load parameters
    mean = tl.load(mean_ptr + feat_idx, mask=mask)
    var = tl.load(var_ptr + feat_idx, mask=mask)
    gamma = tl.load(weight_ptr + feat_idx, mask=mask)
    beta = tl.load(bias_ptr + feat_idx, mask=mask)
    
    # Compute scale and shift
    inv_std = tl.rsqrt(var + eps)
    scale = gamma * inv_std
    shift = beta - mean * scale
    
    # Load input and compute output
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * scale + shift
    
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def bn_optimized(x, running_mean, running_var, weight, bias):
    """
    Optimized batch normalization using Triton kernel.
    """
    n_elements = x.numel()
    features = x.shape[1]
    
    out = torch.empty_like(x)
    
    # Use block size optimized for these tensor sizes
    BLOCK_SIZE = 512
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    bn_kernel[grid](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        n_elements,
        features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return bn_optimized