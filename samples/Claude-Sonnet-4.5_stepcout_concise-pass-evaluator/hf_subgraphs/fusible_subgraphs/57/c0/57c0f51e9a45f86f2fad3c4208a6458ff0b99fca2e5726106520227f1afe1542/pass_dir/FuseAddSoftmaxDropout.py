import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern: add + float + softmax + type_as + dropout (training=False)
    """
    in_1 += in_0
    tmp_0 = in_1
    tmp_1 = tmp_0.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(tmp_0)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return (tmp_4,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_add_softmax_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    M,  # Number of rows
    N,  # Size of last dimension (for softmax)
    stride_in0_batch,
    stride_in0_row,
    stride_in1_batch,
    stride_in1_row,
    stride_out_batch,
    stride_out_row,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row for softmax
    row_idx = tl.program_id(0)
    
    if row_idx >= M:
        return
    
    # Calculate base pointers for this row
    in_0_row_ptr = in_0_ptr + (row_idx // stride_in0_row) * stride_in0_batch + (row_idx % stride_in0_row) * N
    in_1_row_ptr = in_1_ptr + (row_idx // stride_in1_row) * stride_in1_batch + (row_idx % stride_in1_row) * N
    out_row_ptr = out_ptr + (row_idx // stride_out_row) * stride_out_batch + (row_idx % stride_out_row) * N
    
    # Load data in blocks and compute max for numerical stability
    max_val = float('-inf')
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        in_0_vals = tl.load(in_0_row_ptr + offsets, mask=mask, other=float('-inf'))
        in_1_vals = tl.load(in_1_row_ptr + offsets, mask=mask, other=float('-inf'))
        
        # Add
        vals = in_0_vals + in_1_vals
        
        # Find max
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # Compute sum of exp(x - max)
    sum_exp = 0.0
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        in_0_vals = tl.load(in_0_row_ptr + offsets, mask=mask, other=0.0)
        in_1_vals = tl.load(in_1_row_ptr + offsets, mask=mask, other=0.0)
        
        # Add
        vals = in_0_vals + in_1_vals
        
        # Compute exp(x - max)
        exp_vals = tl.exp(vals - max_val)
        
        # Sum
        block_sum = tl.sum(tl.where(mask, exp_vals, 0.0), axis=0)
        sum_exp += block_sum
    
    # Compute softmax and store
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        in_0_vals = tl.load(in_0_row_ptr + offsets, mask=mask, other=0.0)
        in_1_vals = tl.load(in_1_row_ptr + offsets, mask=mask, other=0.0)
        
        # Add
        vals = in_0_vals + in_1_vals
        
        # Softmax: exp(x - max) / sum
        softmax_vals = tl.exp(vals - max_val) / sum_exp
        
        # Store result
        tl.store(out_row_ptr + offsets, softmax_vals, mask=mask)

@torch.fx.wrap
def fused_add_softmax(in_0, in_1):
    # Get original dtype
    orig_dtype = in_1.dtype
    
    # Get shape information
    shape = in_1.shape
    M = 1
    for i in range(len(shape) - 1):
        M *= shape[i]
    N = shape[-1]
    
    # Create output tensor
    out = torch.empty_like(in_1, dtype=torch.float32)
    
    # Calculate strides for proper indexing
    stride_in0_batch = in_0.stride(0) if len(in_0.shape) > 1 else 0
    stride_in0_row = in_0.stride(-2) if len(in_0.shape) > 1 else 1
    stride_in1_batch = in_1.stride(0) if len(in_1.shape) > 1 else 0
    stride_in1_row = in_1.stride(-2) if len(in_1.shape) > 1 else 1
    stride_out_batch = out.stride(0) if len(out.shape) > 1 else 0
    stride_out_row = out.stride(-2) if len(out.shape) > 1 else 1
    
    # Launch kernel
    grid = (M,)
    fused_add_softmax_kernel[grid](
        in_0,
        in_1,
        out,
        M,
        N,
        stride_in0_batch,
        stride_in0_row,
        stride_in1_batch,
        stride_in1_row,
        stride_out_batch,
        stride_out_row,
    )
    
    # Convert back to original dtype (type_as)
    out = out.to(orig_dtype)
    
    # Dropout with training=False is a no-op, so we just return
    return (out,)

def replacement_func():
    return fused_add_softmax