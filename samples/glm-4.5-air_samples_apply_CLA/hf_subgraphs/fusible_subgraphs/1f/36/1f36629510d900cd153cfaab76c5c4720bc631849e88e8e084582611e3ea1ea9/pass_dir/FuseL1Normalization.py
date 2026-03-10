import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = x.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    x /= tmp_1
    return x

def replacement_args(x):
    return (x,)

@triton.jit
def l1_norm_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Calculate row offset and column offsets
    row_offset = row_idx * n_cols
    col_offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    
    # Load row data
    mask = col_offsets < (row_idx + 1) * n_cols
    x = tl.load(x_ptr + col_offsets, mask=mask, other=0.0)
    
    # Compute sum for this row
    row_sum = tl.sum(x)
    
    # Load entire row for normalization (broadcast sum)
    normalized_x = x / row_sum
    
    # Store result
    tl.store(out_ptr + col_offsets, normalized_x, mask=mask)

@torch.fx.wrap
def fused_l1_norm(x):
    n_rows, n_cols = x.shape[-2:]
    N = x.numel()
    BLOCK_SIZE = min(1024, n_cols)
    num_rows = (n_rows + 31) // 32  # Use blocks of 32 rows for better occupancy
    num_cols = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    l1_norm_kernel[(num_rows, num_cols)](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_l1_norm