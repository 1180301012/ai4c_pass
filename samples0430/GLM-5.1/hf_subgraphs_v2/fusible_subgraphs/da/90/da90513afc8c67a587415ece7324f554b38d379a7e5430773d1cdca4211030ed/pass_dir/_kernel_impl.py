import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_layernorm_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr, out_ptr,
    n_rows, n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused add + layer_norm kernel - optimized single-pass variance."""
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    
    # Load x and y, cast to float32 for numerical stability
    x = tl.load(x_ptr + row_start + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row_start + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    
    # Fused add: z = x + y (no intermediate storage)
    z = x + y
    
    # Single-pass mean and variance computation
    # E[x] = sum(x)/n, Var = E[x^2] - E[x]^2
    inv_n_cols = 1.0 / n_cols
    z_sum = tl.sum(z, axis=0) * inv_n_cols  # mean
    z_sq_mean = tl.sum(z * z, axis=0) * inv_n_cols  # E[x^2]
    var = z_sq_mean - z_sum * z_sum  # Var = E[x^2] - E[x]^2
    
    # Normalize: (z - mean) / sqrt(var + eps)
    rstd = 1.0 / tl.sqrt(var + eps)
    # For padded positions, z=0 so (z-mean)*rstd = -mean*rstd, but we mask the store
    z_norm = (z - z_sum) * rstd
    
    # Apply affine transform: out = z_norm * weight + bias
    w = tl.load(w_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    out = z_norm * w + b
    
    # Store result (Triton auto-casts float32 to output dtype)
    tl.store(out_ptr + row_start + col_offsets, out, mask=col_mask)


@torch.fx.wrap
def fused_add_layernorm_dispatch(bias, weight, x, y, route="default"):
    """Shared dispatch wrapper for all fused add + layer_norm passes."""
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Set num_warps based on workload size for better GPU utilization
    if BLOCK_SIZE >= 512:
        num_warps = 8
    elif BLOCK_SIZE >= 256:
        num_warps = 4
    else:
        num_warps = 2
    
    grid = (n_rows,)
    fused_add_layernorm_kernel[grid](
        x_ptr=x, y_ptr=y, w_ptr=weight, b_ptr=bias, out_ptr=out,
        n_rows=n_rows, n_cols=n_cols,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return out