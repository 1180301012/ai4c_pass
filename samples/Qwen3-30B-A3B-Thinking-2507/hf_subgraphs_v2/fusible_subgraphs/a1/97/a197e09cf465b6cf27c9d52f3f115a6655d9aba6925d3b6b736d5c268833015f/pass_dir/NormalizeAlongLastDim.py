import torch
import triton
import triton.language as tl

def pattern(in_0):
    s = in_0.sum(dim=-1)
    s_unsq = s.unsqueeze(-1)
    out = in_0 / s_unsq
    return out

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def sum_reduce_kernel(
    in_ptr,
    sum_ptr,
    B,
    H,
    Y,
    X,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= B * H * Y:
        return
    offset = row_idx * X
    row = tl.load(in_ptr + offset + tl.arange(0, X), mask=tl.arange(0, X) < X, other=0.0)
    s = tl.sum(row, axis=0)
    tl.store(sum_ptr + row_idx, s)

@triton.jit
def div_kernel(
    in_ptr,
    sum_ptr,
    out_ptr,
    B_H_Y,
    X,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offset = tl.program_id(1) * BLOCK_SIZE
    col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
    mask = col_idx < X
    sum_val = tl.load(sum_ptr + row_idx)
    in_offset = row_idx * X + col_offset
    in_vals = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    out_vals = in_vals / sum_val
    tl.store(out_ptr + in_offset, out_vals, mask=mask)

@torch.fx.wrap
def normalize_kernel_wrapper(in_0):
    B, H, Y, X = in_0.shape
    B_H_Y = B * H * Y
    sum_tensor = torch.empty((B_H_Y,), device=in_0.device, dtype=in_0.dtype)
    
    num_sum_blocks = (B_H_Y + 128 - 1) // 128
    sum_reduce_kernel[num_sum_blocks, 128](
        in_0, sum_tensor, B, H, Y, X, 128
    )
    
    out_tensor = torch.empty_like(in_0)
    num_div_blocks_x = (X + 128 - 1) // 128
    div_kernel[(B_H_Y, num_div_blocks_x), 128](
        in_0, sum_tensor, out_tensor, B_H_Y, X, 128
    )
    return out_tensor

def replacement_func():
    return normalize_kernel_wrapper