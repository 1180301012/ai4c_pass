import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Pattern to match: add + split + permute + view operations for 577x384 (24x24 patches)"""
    tmp_0 = in_1 + in_0
    tmp_1 = torch.functional.split(tmp_0, [1, 576], 1)
    tmp_2 = tmp_1[0]
    tmp_3 = tmp_1[1]
    tmp_4 = tmp_3.permute(0, 2, 1)
    tmp_5 = tmp_4.view(1, 384, 24, 24)
    return (tmp_2, tmp_5)


def replacement_args(in_0, in_1):
    """Extract arguments for replacement"""
    return (in_0, in_1)


@triton.jit
def fused_add_split_permute_kernel_24x24(
    in0_ptr,
    in1_ptr,
    out1_ptr,  # First output (class token)
    out2_ptr,  # Second output (reshaped patches)
    batch_size,
    seq_len,
    hidden_dim,
    split_size1,
    split_size2,
    out2_h,
    out2_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for add + split + permute + view
    - out1: shape [batch, 1, hidden_dim] - class token
    - out2: shape [batch, hidden_dim, h, w] - reshaped patches after permute
    """
    pid = tl.program_id(0)
    
    # Total elements to process
    total_elements = batch_size * seq_len * hidden_dim
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load and add
    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    added = in0 + in1
    
    # Determine which split region each element belongs to
    # Original shape: [batch, seq_len, hidden_dim]
    # We need to compute batch, seq, hidden indices for each offset
    
    batch_idx = offsets // (seq_len * hidden_dim)
    remainder = offsets % (seq_len * hidden_dim)
    seq_idx = remainder // hidden_dim
    hidden_idx = remainder % hidden_dim
    
    # Split at dimension 1 (seq_len): [split_size1, split_size2]
    # First split: seq_idx < split_size1
    is_first_split = seq_idx < split_size1
    
    # For first split (class token): store directly
    # Shape: [batch, 1, hidden_dim]
    out1_offset = batch_idx * split_size1 * hidden_dim + (seq_idx - 0) * hidden_dim + hidden_idx
    tl.store(out1_ptr + out1_offset, added, mask=mask & is_first_split)
    
    # For second split: need to permute (0, 2, 1) then view
    # Original: [batch, split_size2, hidden_dim]
    # After permute: [batch, hidden_dim, split_size2]
    # After view: [batch, hidden_dim, h, w]
    
    # Adjusted seq index for second split
    seq_idx_adj = seq_idx - split_size1
    
    # Permute (0, 2, 1): swap seq and hidden dimensions
    # New indices: batch stays same, hidden_dim becomes dim1, seq_idx_adj becomes dim2
    # Linear offset in permuted space: batch * (hidden_dim * split_size2) + hidden_idx * split_size2 + seq_idx_adj
    out2_offset = batch_idx * (hidden_dim * split_size2) + hidden_idx * split_size2 + seq_idx_adj
    
    tl.store(out2_ptr + out2_offset, added, mask=mask & (~is_first_split))


@torch.fx.wrap
def fused_add_split_permute_view_24x24(in_0, in_1):
    """Wrapper function for the fused kernel - 577x384 variant"""
    batch_size, seq_len, hidden_dim = in_0.shape
    
    # Hardcoded for this variant
    split_size1 = 1
    split_size2 = 576
    h, w = 24, 24
    
    # Allocate outputs
    out1 = torch.empty((batch_size, split_size1, hidden_dim), device=in_0.device, dtype=in_0.dtype)
    out2 = torch.empty((batch_size, hidden_dim, h, w), device=in_0.device, dtype=in_0.dtype)
    
    # Launch kernel
    total_elements = batch_size * seq_len * hidden_dim
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_add_split_permute_kernel_24x24[grid](
        in_0,
        in_1,
        out1,
        out2,
        batch_size,
        seq_len,
        hidden_dim,
        split_size1,
        split_size2,
        h,
        w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out1, out2)


def replacement_func():
    """Return the replacement function"""
    return fused_add_split_permute_view_24x24