import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern to match: just add + softmax (simpler pattern)
    """
    tmp_0 = in_0 + in_1
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def fused_add_softmax_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    N,  # Size of the last dimension
    stride_batch,
    stride_head,
    stride_row,
    stride_col,
    stride_in1_batch,
    stride_in1_head,
    stride_in1_row,
    stride_in1_col,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the softmax
    row_idx = tl.program_id(0)
    
    # Calculate base offset for this row
    batch_idx = row_idx // (stride_batch // stride_row)
    remaining = row_idx % (stride_batch // stride_row)
    head_idx = remaining // (stride_head // stride_row)
    row_in_head = remaining % (stride_head // stride_row)
    
    row_offset = row_idx * stride_col
    in1_row_offset = (batch_idx * stride_in1_batch + 
                      head_idx * stride_in1_head + 
                      row_in_head * stride_in1_row)
    
    # Load and compute in blocks
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # First pass: compute max for numerical stability
    max_val = float('-inf')
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < N
        
        in0_vals = tl.load(in0_ptr + row_offset + offsets, mask=mask, other=float('-inf'))
        in1_vals = tl.load(in1_ptr + in1_row_offset + offsets * stride_in1_col, mask=mask, other=0.0)
        
        vals = in0_vals + in1_vals
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # Second pass: compute sum of exp(x - max)
    sum_exp = 0.0
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < N
        
        in0_vals = tl.load(in0_ptr + row_offset + offsets, mask=mask, other=0.0)
        in1_vals = tl.load(in1_ptr + in1_row_offset + offsets * stride_in1_col, mask=mask, other=0.0)
        
        vals = in0_vals + in1_vals
        exp_vals = tl.exp(vals - max_val)
        block_sum = tl.sum(tl.where(mask, exp_vals, 0.0), axis=0)
        sum_exp += block_sum
    
    # Third pass: compute final softmax values and store
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < N
        
        in0_vals = tl.load(in0_ptr + row_offset + offsets, mask=mask, other=0.0)
        in1_vals = tl.load(in1_ptr + in1_row_offset + offsets * stride_in1_col, mask=mask, other=0.0)
        
        vals = in0_vals + in1_vals
        exp_vals = tl.exp(vals - max_val)
        softmax_vals = exp_vals / sum_exp
        
        tl.store(out_ptr + row_offset + offsets, softmax_vals, mask=mask)


@torch.fx.wrap
def fused_add_softmax(in_0, in_1):
    """
    Fused implementation of add + softmax
    Handles broadcasting of in_1
    """
    # Ensure inputs are contiguous
    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()
    
    # Get shapes
    shape = in_0.shape
    N = shape[-1]  # Last dimension for softmax
    
    # Calculate total number of rows
    num_rows = in_0.numel() // N
    
    # Prepare output
    out = torch.empty_like(in_0)
    
    # Calculate strides
    stride_batch = in_0.stride(0) if in_0.dim() > 0 else 0
    stride_head = in_0.stride(1) if in_0.dim() > 1 else 0
    stride_row = in_0.stride(2) if in_0.dim() > 2 else 0
    stride_col = in_0.stride(3) if in_0.dim() > 3 else 1
    
    # Handle broadcasting for in_1
    stride_in1_batch = in_1.stride(0) if in_1.dim() > 0 and in_1.shape[0] > 1 else 0
    stride_in1_head = in_1.stride(1) if in_1.dim() > 1 and in_1.shape[1] > 1 else 0
    stride_in1_row = in_1.stride(2) if in_1.dim() > 2 and in_1.shape[2] > 1 else 0
    stride_in1_col = in_1.stride(3) if in_1.dim() > 3 else (in_1.stride(-1) if in_1.dim() > 0 else 0)
    
    # Launch kernel
    grid = (num_rows,)
    
    fused_add_softmax_kernel[grid](
        in_0,
        in_1,
        out,
        N,
        stride_batch,
        stride_head,
        stride_row,
        stride_col,
        stride_in1_batch,
        stride_in1_head,
        stride_in1_row,
        stride_in1_col,
    )
    
    return out


def replacement_func():
    return fused_add_softmax