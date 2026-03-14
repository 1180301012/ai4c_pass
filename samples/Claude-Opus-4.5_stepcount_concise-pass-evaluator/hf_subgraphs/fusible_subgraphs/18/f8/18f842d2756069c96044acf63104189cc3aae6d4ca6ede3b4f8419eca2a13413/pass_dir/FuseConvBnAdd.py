import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias):
    """
    Pattern to match: batch_norm
    """
    bn_out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return bn_out


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized batch normalization kernel"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Compute channel index: for NCHW layout
    # offset = n*C*HW + c*HW + hw
    # c = (offset // HW) % C
    c_idx = (offsets // HW) % C
    
    # Load BN parameters
    mean = tl.load(running_mean_ptr + c_idx, mask=mask)
    var = tl.load(running_var_ptr + c_idx, mask=mask)
    gamma = tl.load(weight_ptr + c_idx, mask=mask)
    beta = tl.load(bias_ptr + c_idx, mask=mask)
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute batch norm: y = gamma * (x - mean) * rsqrt(var + eps) + beta
    inv_std = tl.math.rsqrt(var + eps)
    out = gamma * (x - mean) * inv_std + beta
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def bn_fused(x, running_mean, running_var, weight, bias):
    """Optimized batch norm implementation"""
    N, C, H, W = x.shape
    HW = H * W
    n_elements = N * C * HW
    
    out = torch.empty_like(x)
    
    # Launch kernel
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    batch_norm_kernel[grid](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        C,
        HW,
        n_elements,
        1e-05,
    )
    
    return out


def replacement_func():
    return bn_fused