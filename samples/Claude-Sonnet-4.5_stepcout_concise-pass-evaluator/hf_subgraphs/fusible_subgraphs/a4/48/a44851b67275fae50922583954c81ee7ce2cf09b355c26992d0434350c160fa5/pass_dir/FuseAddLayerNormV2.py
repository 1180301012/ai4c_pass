import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """Pattern: add followed by layer_norm for convit_small (432 dims)"""
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (432,), in_1, in_0, 1e-06)
    return tmp_2, tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_add_layer_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    out_add_ptr,
    out_norm_ptr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for add + layer_norm"""
    row_idx = tl.program_id(0)
    
    # Pointers to the current row
    x_row_ptr = x_ptr + row_idx * N
    y_row_ptr = y_ptr + row_idx * N
    out_add_row_ptr = out_add_ptr + row_idx * N
    out_norm_row_ptr = out_norm_ptr + row_idx * N
    
    # Compute mean and variance
    mean = 0.0
    var = 0.0
    
    # First pass: compute add and mean
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_row_ptr + offsets, mask=mask, other=0.0)
        
        added = x + y
        
        # Store added result
        tl.store(out_add_row_ptr + offsets, added, mask=mask)
        
        # Accumulate mean
        mean += tl.sum(tl.where(mask, added, 0.0))
    
    mean = mean / N
    
    # Second pass: compute variance
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        added = tl.load(out_add_row_ptr + offsets, mask=mask, other=0.0)
        
        # Accumulate variance
        diff = added - mean
        var += tl.sum(tl.where(mask, diff * diff, 0.0))
    
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Third pass: normalize and apply affine transformation
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        added = tl.load(out_add_row_ptr + offsets, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
        bias_val = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        
        # Normalize
        normalized = (added - mean) * rstd
        
        # Apply affine transformation
        out = normalized * weight + bias_val
        
        tl.store(out_norm_row_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_layer_norm(bias, weight, x, y):
    """Wrapper for fused add + layer_norm kernel"""
    batch_size, seq_len, hidden_dim = x.shape
    M = batch_size * seq_len
    N = hidden_dim
    
    # Flatten batch and sequence dimensions
    x_flat = x.reshape(M, N).contiguous()
    y_flat = y.reshape(M, N).contiguous()
    
    # Allocate output tensors
    out_add_flat = torch.empty((M, N), dtype=x.dtype, device=x.device)
    out_norm_flat = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    grid = (M,)
    fused_add_layer_norm_kernel[grid](
        x_flat,
        y_flat,
        weight,
        bias,
        out_add_flat,
        out_norm_flat,
        M,
        N,
        1e-06,
    )
    
    # Reshape outputs
    out_add = out_add_flat.reshape(batch_size, seq_len, hidden_dim)
    out_norm = out_norm_flat.reshape(batch_size, seq_len, hidden_dim)
    
    return out_add, out_norm


def replacement_func():
    return fused_add_layer_norm