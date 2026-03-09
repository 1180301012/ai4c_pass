import torch
import triton
import triton.language as tl
import math

def pattern(in_0):
    tmp_0 = in_0
    tmp_1 = tmp_0.ne(1)
    tmp_0 = None
    tmp_2 = tmp_1.int()
    tmp_1 = None
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_3 = None
    tmp_5 = tmp_4 + 0
    tmp_4 = None
    tmp_6 = tmp_5 * tmp_2
    tmp_5 = tmp_2 = None
    tmp_7 = tmp_6.long()
    tmp_6 = None
    tmp_8 = tmp_7 + 1
    tmp_7 = None
    return (tmp_8,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def cumsum_mask_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Compute base address for this batch
    x_base = batch_idx * seq_len  
    out_base = batch_idx * seq_len
    
    # Load the current element
    current_idx = x_base + seq_idx
    x_val = tl.load(x_ptr + current_idx)
    
    # Check if element != 1 (same as in_0.ne(1))
    is_non_one = (x_val != 1)
    
    # Compute cumulative sum - handle first element separately
    if seq_idx == 0:
        # For first element: if non-one, output 1, otherwise 0
        final_val = tl.cast(is_non_one, tl.int64)  # True->1, False->0, both as int64
    else:
        # For subsequent elements: read previous cumulative count and add 1 if current is non-one
        prev_cumsum = tl.load(out_ptr + out_base + seq_idx - 1)
        add_value = tl.cast(is_non_one, tl.int64)  # True->1, False->0, both as int64
        final_val = prev_cumsum + add_value
    
    # Store the result
    tl.store(out_ptr + out_base + seq_idx, final_val)

@torch.fx.wrap
def optimized_cumsum_mask(in_0):
    # Get tensor properties
    batch_size, seq_len = in_0.shape
    num_elements = batch_size * seq_len
    
    # Choose optimal block size based on sequence length for better GPU utilization
    if seq_len <= 64:
        BLOCK_SIZE = 64
    elif seq_len <= 256:
        BLOCK_SIZE = 128  
    elif seq_len <= 512:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = min(1024, seq_len)
    
    # Calculate grid dimensions - each program handles one element efficiently
    grid = (batch_size, seq_len)
    
    # Create output tensor
    out = torch.empty_like(in_0, dtype=torch.int64)
    
    # Launch kernel with autotuning
    cumsum_mask_kernel[grid](
        in_0,
        out,
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_cumsum_mask