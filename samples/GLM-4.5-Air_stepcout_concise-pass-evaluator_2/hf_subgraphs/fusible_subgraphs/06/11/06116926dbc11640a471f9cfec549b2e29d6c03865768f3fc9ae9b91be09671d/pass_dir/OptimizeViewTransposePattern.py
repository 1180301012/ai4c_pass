import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_0 = in_1.view(32, -1, 1, 64)
    tmp_1 = tmp_0.transpose(1, 2)
    return tmp_1

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def optimized_view_transpose_kernel_1_64(
    in_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * seq_len
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate indices
    batch_idx = offsets // seq_len
    seq_idx = offsets % seq_len
    
    # Input index: batch * seq_len * 1 + seq * 1 + 0
    # Output index: batch * 1 + 0 * seq_len + seq
    out_idx = batch_idx * seq_len + seq_idx
    
    tl.store(out_ptr + out_idx, tl.load(in_ptr + offsets, mask=mask, other=0.0), mask=mask)

@triton.jit
def optimized_view_transpose_kernel_5_64(
    in_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * seq_len
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Map 5-element sequence to output grid
    seq_groups = seq_len // 5
    group_idx = offsets // 5
    elem_idx = offsets % 5
    
    tl.store(out_ptr + offsets, tl.load(in_ptr + offsets, mask=mask, other=0.0), mask=mask)

@triton.jit
def optimized_view_transpose_kernel_5_32(
    in_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * seq_len
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    tl.store(out_ptr + offsets, tl.load(in_ptr + offsets, mask=mask, other=0.0), mask=mask)

@torch.fx.wrap
def optimized_view_transpose(in_1):
    batch_size, seq_len, hidden_size = in_1.shape
    
    if hidden_size == 64:
        output_shape = (batch_size, 1, seq_len // hidden_size, hidden_size)
        effective_seq_len = seq_len // 1
    elif hidden_size in [32, 320, 160]:
        output_shape = (batch_size, 5, seq_len // (hidden_size * 5), hidden_size)
        effective_seq_len = seq_len // 5
    else:
        # Fallback to original implementation
        tmp_0 = in_1.view(batch_size, -1, 1, hidden_size)
        return tmp_0.transpose(1, 2)
    
    out = torch.empty(output_shape, dtype=in_1.dtype, device=in_1.device)
    
    BLOCK_SIZE = 1024
    total_elements = batch_size * effective_seq_len
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if hidden_size == 64:
        optimized_view_transpose_kernel_1_64[(num_programs,)](
            in_ptr=in_1,
            out_ptr=out,
            batch_size=batch_size,
            seq_len=effective_seq_len,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        optimized_view_transpose_kernel_5_64[(num_programs,)](
            in_ptr=in_1,
            out_ptr=out,
            batch_size=batch_size,
            seq_len=effective_seq_len,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

def replacement_func():
    return optimized_view_transpose