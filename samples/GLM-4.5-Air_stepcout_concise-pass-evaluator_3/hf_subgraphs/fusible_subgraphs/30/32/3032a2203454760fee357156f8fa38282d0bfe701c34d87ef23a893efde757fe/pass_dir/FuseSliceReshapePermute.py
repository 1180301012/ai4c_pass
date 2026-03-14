import torch
import triton
import triton.language as tl
import numpy as np

def pattern(in_0, in_1, in_2, in_3):
    # Match the slice + reshape + permute sequence without affecting tmp_2
    tmp_2 = in_2 + in_3
    # Note: tmp_2 is used by multiple consumers, so we don't modify it
    tmp_3 = tmp_2[slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_4 = tmp_3.reshape(1, 12, 12, -1)
    tmp_3 = None
    out = tmp_4.permute(0, 3, 1, 2)
    tmp_4 = None
    # Return only the permuted result - this intermediate is not used in the final output
    return out

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_slice_reshape_permute_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    orig_seq_len,
    hidden_size,
    new_seq_len,
    BLOCK_SIZE_H: tl.constexpr,
):
    # Each program handles one hidden dimension element
    pid = tl.program_id(0)
    
    # Calculate hidden dimension and position in new sequence
    hidden_idx = pid
    seq_idx = (pid // hidden_size) if pid < hidden_size * new_seq_len else 0
    
    # Check bounds
    if hidden_idx >= hidden_size or seq_idx >= new_seq_len:
        return
    
    # Calculate original sequence index (skip first element)
    orig_seq_idx = seq_idx + 1
    
    # Calculate offsets
    x_offset = (batch_size * orig_seq_len * hidden_size) + (orig_seq_idx * hidden_size) + hidden_idx
    y_offset = (batch_size * orig_seq_len * hidden_size) + (orig_seq_idx * hidden_size) + hidden_idx
    
    # Load both tensors
    x_loaded = tl.load(x_ptr + x_offset, other=0.0)
    y_loaded = tl.load(y_ptr + y_offset, other=0.0)
    
    # Add operation
    added = x_loaded + y_loaded
    
    # Store result directly in permuted format: (batch, hidden, 12, 12)
    # Convert flat sequence index to 2D grid position (12x12)
    grid_i = seq_idx // 12
    grid_j = seq_idx % 12
    out_offset = (batch_size * hidden_size * 12 * 12) + (hidden_idx * 12 * 12) + (grid_i * 12) + grid_j
    
    tl.store(out_ptr + out_offset, added)

@torch.fx.wrap
def fused_slice_reshape_permute_optimized(in_0, in_1, in_2, in_3):
    # Input tensors are [1, 145, 512] for in_2, in_3
    # in_0, in_1 are unused in the computation but preserved for the pattern
    
    # Optimize the slice-reshape-permute sequence
    x, y = in_2, in_3
    batch_size, original_seq_len, hidden_size = x.shape
    
    # We want to remove first element and reshape to [1, 512, 12, 12]
    # where 12x12 = 144, accounting for the slice
    new_seq_len = original_seq_len - 1  # 144 = 12*12
    
    out = torch.empty(1, hidden_size, 12, 12, dtype=x.dtype, device=x.device)
    
    # Calculate grid size - each program handles one hidden dimension element
    total_elements = hidden_size * new_seq_len
    BLOCK_SIZE_H = hidden_size  # Process one full hidden dimension per program
    
    # Launch grid for all elements
    grid = (total_elements,)
    
    fused_slice_reshape_permute_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out.view(-1).contiguous(),  # Flatten for easier indexing
        batch_size=batch_size,
        orig_seq_len=original_seq_len,
        hidden_size=hidden_size,
        new_seq_len=new_seq_len,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    
    return out

def replacement_func():
    return fused_slice_reshape_permute_optimized