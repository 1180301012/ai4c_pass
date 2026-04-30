import torch
import triton
import triton.language as tl

@triton.jit
def fused_max_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    pid = tl.program_id(0)
    
    # Compute batch and seq indices
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Compute row offset
    row_offset = batch_idx * row_stride + seq_idx * n_cols
    
    # Load values
    col_offsets = tl.arange(0, BLOCK_SIZE)
    offsets = row_offset + col_offsets
    mask = col_offsets < n_cols
    vals = tl.load(input_ptr + offsets, mask=mask, other=float("-inf"))
    
    # Compute max along last dimension
    max_val = tl.max(vals)
    
    # Broadcast max back to full shape
    tl.store(output_ptr + offsets, max_val, mask=mask)


@torch.fx.wrap
def fused_max_wrapper(in_0):
    batch_size, seq_len, n_cols = in_0.shape
    out = torch.empty_like(in_0)
    
    # Launch one program per row
    n_rows = batch_size * seq_len
    grid = (n_rows,)
    
    fused_max_kernel[grid](
        in_0, out, 
        batch_size, seq_len, 
        in_0.stride(1) * n_cols,  # row_stride in elements
        n_cols,
        BLOCK_SIZE=512
    )
    return out


def pattern(in_0):
    return torch.max(in_0, -1, keepdim=True)


def replacement_args(in_0):
    return in_0,


def replacement_func():
    return fused_max_wrapper