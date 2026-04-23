import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Match the computation pattern:
    cumsum + multiply + subtract + cast + slice + add
    """
    tmp_0 = in_0
    tmp_1 = torch.cumsum(tmp_0, dim=1)
    tmp_2 = tmp_1 * tmp_0
    tmp_3 = tmp_2 - 1
    tmp_4 = tmp_3.long()
    tmp_5 = tmp_4[slice(None, None, None), slice(0, None, None)]
    tmp_6 = tmp_5 + 2
    return tmp_6


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_kernel(in_ptr, out_ptr, n_cols: tl.constexpr, n_rows: tl.constexpr):
    """
    Fused kernel: cumsum -> multiply -> subtract -> cast -> add
    
    Single program per row, processing all columns.
    Uses BLOCK_SIZE=16 (power of 2 >= max n_cols=13).
    """
    pid = tl.program_id(0)
    
    # Load row with mask
    col_offsets = tl.arange(0, 16)
    col_mask = col_offsets < n_cols
    
    x = tl.load(in_ptr + pid * n_cols + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    
    # Compute cumsum along dim=1 (columns)
    cs = tl.cumsum(x)
    
    # Fused: (cumsum * input) - 1 + 2 = cumsum * input + 1
    result = (cs * x + 1.0).to(tl.int64)
    
    # Store
    tl.store(out_ptr + pid * n_cols + col_offsets, result, mask=col_mask)


@torch.fx.wrap
def wrapper(in_0):
    n_rows, n_cols = in_0.shape
    out = torch.empty_like(in_0)
    
    # Launch one program per row
    fused_kernel[(n_rows,)](in_ptr=in_0, out_ptr=out, n_cols=n_cols, n_rows=n_rows)
    
    return out


def replacement_func():
    return wrapper