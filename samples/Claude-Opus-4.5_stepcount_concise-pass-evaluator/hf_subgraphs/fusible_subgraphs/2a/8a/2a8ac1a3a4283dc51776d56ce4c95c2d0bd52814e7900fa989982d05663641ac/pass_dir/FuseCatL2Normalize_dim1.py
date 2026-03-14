import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Pattern to match: cat single tensor + L2 normalize
    """
    tmp_0 = torch.cat([in_0], 1)
    tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 768}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 768}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def l2_normalize_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    x_stride_row,
    out_stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """
    L2 normalize each row of a 2D tensor.
    Each program handles one row.
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Pointer to the start of this row
    x_row_ptr = x_ptr + row_idx * x_stride_row
    out_row_ptr = out_ptr + row_idx * out_stride_row
    
    # First pass: compute sum of squares
    sum_sq = tl.zeros([1], dtype=tl.float32)
    
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x_vals = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
    
    # Compute norm (with epsilon for numerical stability)
    eps = 1e-12
    norm = tl.sqrt(sum_sq + eps)
    
    # Second pass: normalize and store
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x_vals = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)
        out_vals = x_vals / norm
        tl.store(out_row_ptr + col_offsets, out_vals, mask=mask)


@torch.fx.wrap
def fused_cat_l2_normalize(x):
    """
    Fused cat + L2 normalize implementation using Triton.
    Since cat([x], 1) is a no-op, we just do L2 normalization.
    """
    # Ensure contiguous
    x = x.contiguous()
    
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    
    # Launch one program per row
    grid = (n_rows,)
    
    l2_normalize_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        x_stride_row=x.stride(0),
        out_stride_row=out.stride(0),
    )
    
    return out


def replacement_func():
    return fused_cat_l2_normalize