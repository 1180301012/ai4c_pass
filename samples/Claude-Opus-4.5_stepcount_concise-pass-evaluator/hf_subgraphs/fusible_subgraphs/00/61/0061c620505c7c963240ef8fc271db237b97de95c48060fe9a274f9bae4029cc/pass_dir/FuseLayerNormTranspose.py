import torch
import triton
import triton.language as tl


def pattern(bias, weight, x):
    """
    Pattern to match layer_norm followed by transpose(-1, -2)
    """
    ln_out = torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-05)
    transposed = ln_out.transpose(-1, -2)
    return transposed


def replacement_args(bias, weight, x):
    return (bias, weight, x)


@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    N_SIZE: tl.constexpr,
):
    """
    Layer Norm kernel optimized for N=768
    Each program handles one row
    """
    # Program ID is the row index
    row_idx = tl.program_id(0)
    
    # Compute offset for this row
    row_start = row_idx * N_SIZE
    
    # Load offsets
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < N_SIZE
    
    # Load the entire row
    x = tl.load(x_ptr + row_start + n_offsets, mask=n_mask, other=0.0).to(tl.float32)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + n_offsets, mask=n_mask, other=1.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0).to(tl.float32)
    
    # Compute mean using Var(X) = E[X^2] - E[X]^2 approach
    x_masked = tl.where(n_mask, x, 0.0)
    x_sum = tl.sum(x_masked, axis=0)
    mean = x_sum * (1.0 / N_SIZE)
    
    # Compute variance
    x_sq = tl.where(n_mask, x * x, 0.0)
    x_sq_sum = tl.sum(x_sq, axis=0)
    mean_sq = mean * mean
    var = x_sq_sum * (1.0 / N_SIZE) - mean_sq
    
    # Normalize and apply affine transform
    rstd = tl.rsqrt(var + eps)
    x_norm = (x - mean) * rstd
    out = x_norm * weight + bias_val
    
    # Write output
    tl.store(out_ptr + row_start + n_offsets, out, mask=n_mask)


@torch.fx.wrap
def fused_layer_norm_transpose(bias, weight, x):
    """
    Fused layer norm + transpose
    Input x: [B, M, N]
    Output: [B, N, M] (as a view)
    """
    # Get dimensions
    B = x.shape[0]
    M = x.shape[1]
    N = x.shape[2]
    
    # Allocate output with same shape as input
    ln_out = torch.empty((B, M, N), device=x.device, dtype=x.dtype)
    
    # Launch kernel
    num_rows = B * M
    grid = (num_rows,)
    
    layer_norm_kernel[grid](
        x,
        weight,
        bias,
        ln_out,
        1e-05,
        BLOCK_SIZE_N=1024,
        N_SIZE=768,
        num_warps=4,
    )
    
    # Return transposed view (no data copy)
    return ln_out.transpose(-1, -2)


def replacement_func():
    return fused_layer_norm_transpose