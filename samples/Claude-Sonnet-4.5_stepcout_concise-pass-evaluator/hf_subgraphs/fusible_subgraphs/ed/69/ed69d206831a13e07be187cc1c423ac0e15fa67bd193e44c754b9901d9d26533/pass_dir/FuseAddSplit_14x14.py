import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Simpler pattern: just add + split operations for 197x384"""
    tmp_0 = in_1 + in_0
    tmp_1 = torch.functional.split(tmp_0, [1, 196], 1)
    return tmp_1


def replacement_args(in_0, in_1):
    """Extract arguments for replacement"""
    return (in_0, in_1)


@triton.jit
def fused_add_split_kernel(
    in0_ptr,
    in1_ptr,
    out1_ptr,  # First split output  
    out2_ptr,  # Second split output
    batch_size,
    seq_len,
    hidden_dim,
    split_size1,
    split_size2,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for add + split"""
    pid = tl.program_id(0)
    
    total_elements = batch_size * seq_len * hidden_dim
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load and add
    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    added = in0 + in1
    
    # Compute indices
    batch_idx = offsets // (seq_len * hidden_dim)
    remainder = offsets % (seq_len * hidden_dim)
    seq_idx = remainder // hidden_dim
    hidden_idx = remainder % hidden_dim
    
    # Split at dimension 1
    is_first_split = seq_idx < split_size1
    
    # Store to first split
    out1_offset = batch_idx * split_size1 * hidden_dim + seq_idx * hidden_dim + hidden_idx
    tl.store(out1_ptr + out1_offset, added, mask=mask & is_first_split)
    
    # Store to second split
    seq_idx_adj = seq_idx - split_size1
    out2_offset = batch_idx * split_size2 * hidden_dim + seq_idx_adj * hidden_dim + hidden_idx
    tl.store(out2_ptr + out2_offset, added, mask=mask & (~is_first_split))


@torch.fx.wrap
def fused_add_split(in_0, in_1):
    """Wrapper function for fused add + split"""
    batch_size, seq_len, hidden_dim = in_0.shape
    
    split_size1 = 1
    split_size2 = 196
    
    # Allocate outputs
    out1 = torch.empty((batch_size, split_size1, hidden_dim), device=in_0.device, dtype=in_0.dtype)
    out2 = torch.empty((batch_size, split_size2, hidden_dim), device=in_0.device, dtype=in_0.dtype)
    
    # Launch kernel
    total_elements = batch_size * seq_len * hidden_dim
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_add_split_kernel[grid](
        in_0,
        in_1,
        out1,
        out2,
        batch_size,
        seq_len,
        hidden_dim,
        split_size1,
        split_size2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out1, out2)


def replacement_func():
    """Return the replacement function"""
    return fused_add_split