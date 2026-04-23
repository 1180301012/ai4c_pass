import torch
import triton
import triton.language as tl


def pattern(in_3):
    """
    Match sum with keepdim (single output pattern)
    """
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    return tmp_5


def replacement_args(in_3):
    return (in_3,)


@triton.jit
def sum_kernel(
    input_ptr,
    sum_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel that computes sum over dim=3 with keepdim-like behavior.
    """
    row_idx = tl.program_id(0)
    row_offset = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    pointers = input_ptr + row_offset + col_offsets
    
    vals = tl.load(pointers, mask=mask, other=0.0)
    row_sum = tl.sum(vals, axis=0)
    
    # Store sum value
    tl.store(sum_ptr + row_idx, row_sum)


@torch.fx.wrap
def fused_sum(input_tensor):
    """
    Fused implementation of sum(dim=3, keepdim=True).
    Output shape matches keepdim=True: [1, 2, 8, 1]
    """
    n_batch, n_channels, H, W = input_tensor.shape
    
    n_rows = n_batch * n_channels * H
    n_cols = W
    BLOCK_SIZE = 8
    
    # Allocate output tensor with correct shape for keepdim
    sum_keepdim = torch.empty(n_batch, n_channels, H, 1, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel - each program handles one row
    grid = (n_rows,)
    sum_kernel[grid](
        input_tensor,
        sum_keepdim,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return sum_keepdim


def replacement_func():
    return fused_sum