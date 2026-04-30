import torch
from torch import device
import triton
import triton.language as tl

@triton.jit
def fused_cat_ones_kernel(
    indexed_data_ptr,
    indexed_data_row_stride,
    other_data_ptr,
    other_data_row_stride,
    mask_ptr,
    out_cat_ptr,
    out_ones_ptr,
    n_indexed_rows,
    n_other_cols,
    n_mask_true,
    BLOCK_SIZE: tl.constexpr,
):
    # Process rows for concatenation
    pid = tl.program_id(0)
    n_rows = n_indexed_rows + n_other_cols
    
    row_offset = pid * n_rows
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load indexed data row
    indexed_row_offset = pid * indexed_data_row_stride
    indexed_mask = (col_offsets < n_indexed_rows)
    indexed_data = tl.load(indexed_data_ptr + indexed_row_offset + col_offsets * indexed_data_row_stride // n_indexed_rows, 
                           mask=indexed_mask, other=0.0)
    
    # Load other data row
    other_row_offset = pid * other_data_row_stride
    other_mask = (col_offsets >= n_indexed_rows) & (col_offsets < n_rows)
    other_data = tl.load(other_data_ptr + other_row_offset + (col_offsets - n_indexed_rows) * other_data_row_stride // n_other_cols,
                         mask=other_mask, other=0.0)
    
    # Concatenate
    cat_out = tl.where(col_offsets < n_indexed_rows, indexed_data, other_data)
    
    # Store concatenated result
    tl.store(out_cat_ptr + row_offset + col_offsets, cat_out, mask=col_offsets < n_rows)
    
    # Process ones tensor
    ones_pid = tl.program_id(1)
    ones_offset = ones_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ones_mask = ones_offset < n_mask_true
    tl.store(out_ones_ptr + ones_offset, tl.ones(BLOCK_SIZE, dtype=tl.float32), mask=ones_mask)


@torch.fx.wrap
def fused_cat_ones(indexed_data, other_data, mask):
    """
    Fused kernel that:
    1. Concatenates indexed_data with other_data along dim=1
    2. Creates ones tensor of size (other_data.shape[1] + mask.sum())
    """
    assert indexed_data.is_contiguous() or indexed_data.stride() == (indexed_data.shape[1], 1)
    
    n_mask_true = mask.sum().item()
    n_indexed_cols = indexed_data.shape[1]
    n_other_cols = other_data.shape[1]
    n_rows = indexed_data.shape[0]
    
    # Create output tensors
    out_cat = torch.empty((n_rows, n_indexed_cols + n_other_cols), 
                          dtype=indexed_data.dtype, device=indexed_data.device)
    out_ones = torch.empty((n_other_cols + n_mask_true,), 
                           dtype=torch.float32, device='cuda')
    
    # Launch kernels
    BLOCK_SIZE = 1024
    
    # Grid for concatenation (process each row)
    cat_grid = (n_rows,)
    fused_cat_ones_kernel[cat_grid, (1,)](
        indexed_data_ptr=indexed_data,
        indexed_data_row_stride=indexed_data.stride(0),
        other_data_ptr=other_data,
        other_data_row_stride=other_data.stride(0),
        mask_ptr=mask,
        out_cat_ptr=out_cat,
        out_ones_ptr=out_ones,
        n_indexed_rows=n_indexed_cols,
        n_other_cols=n_other_cols,
        n_mask_true=n_mask_true,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_cat, out_ones


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: indexing + sym_size + cat + sym_sum + ones
    Returns: (concatenated_result, ones_tensor)
    """
    tmp_1 = in_0[slice(None, None, None), in_2]
    tmp_2 = torch.ops.aten.sym_size.int(tmp_1, 1)
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    tmp_10 = torch.sym_sum([128, tmp_2])
    tmp_11 = torch.ones((tmp_10,), dtype=torch.float32, device=device(type='cuda'))
    return tmp_9, tmp_11


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_cat_ones