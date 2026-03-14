import torch
import triton
import triton.language as tl

def pattern(tmp_0):
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

def replacement_args(tmp_0):
    return (tmp_0,)

@triton.jit
def cumsum_mask_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles one element across all batches
    if pid >= batch_size * seq_len:
        return
    
    batch_idx = pid // seq_len
    elem_idx = pid % seq_len
    
    # Calculate pointer offset
    offset = batch_idx * seq_len + elem_idx
    
    # Load current element
    x = tl.load(x_ptr + offset)
    
    # Create mask for current element
    current_mask = 1 if x != 1 else 0
    
    # Compute cumulative sum from start to current position
    cumsum_val = 0
    for i in range(elem_idx + 1):
        prev_offset = batch_idx * seq_len + i
        prev_x = tl.load(x_ptr + prev_offset)
        prev_mask = 1 if prev_x != 1 else 0
        cumsum_val += prev_mask
    
    # Apply mask and add 1
    result = cumsum_val * current_mask + 1
    
    # Store result
    tl.store(out_ptr + offset, result)

@torch.fx.wrap
def optimized_cumsum_mask(x):
    # Get input shape
    batch_size, seq_len = x.shape
    
    # Create output tensor
    out = torch.empty_like(x, dtype=torch.long)
    
    # Launch kernel with one program per element
    num_programs = batch_size * seq_len
    cumsum_mask_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=1,  # Each program handles one element
    )
    
    return out

def replacement_func():
    return optimized_cumsum_mask