import torch
import triton
import triton.language as tl

def pattern(in_4, tmp_0):
    tmp_9 = in_4[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
    tmp_10 = in_4[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_11 = tmp_0.tensor_split(2, -1)
    tmp_12 = tmp_11[0]
    tmp_13 = tmp_11[1]
    return (tmp_13, tmp_9, tmp_10, tmp_12)

def replacement_args(in_4, tmp_0):
    return (in_4, tmp_0)

@triton.jit
def slice_split_kernel(
    in4_ptr,
    tmp0_ptr,
    out13_ptr,
    out9_ptr,
    out10_ptr,
    out12_ptr,
    batch_size,
    n_heads,
    seq_len_full,
    seq_len_half,
    feat_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements  
    pid = tl.program_id(0)
    
    # Handle first two outputs (slices from in_4)
    total_elements_slice = batch_size * n_heads * seq_len_half * feat_dim
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements_slice
    
    # Calculate indices for slice operations
    batch = offsets // (n_heads * seq_len_half * feat_dim)
    remainder = offsets % (n_heads * seq_len_half * feat_dim)
    head = remainder // (seq_len_half * feat_dim)
    remainder = remainder % (seq_len_half * feat_dim)
    seq = remainder // feat_dim
    feat = remainder % feat_dim
    
    # Load original indices (for seq_len_full)
    seq_full1 = seq  # For tmp_9: first 256 elements
    seq_full2 = seq + 1  # For tmp_10: elements 1-257
    
    # Load from in_4 for both slices
    in4_idx1 = batch * n_heads * seq_len_full * feat_dim + head * seq_len_full * feat_dim + seq_full1 * feat_dim + feat
    in4_idx2 = batch * n_heads * seq_len_full * feat_dim + head * seq_len_full * feat_dim + seq_full2 * feat_dim + feat
    
    in4_val1 = tl.load(in4_ptr + in4_idx1, mask=offsets < total_elements_slice, other=0.0)
    in4_val2 = tl.load(in4_ptr + in4_idx2, mask=offsets < total_elements_slice, other=0.0)
    
    # Store slice results
    tl.store(out9_ptr + offsets, in4_val1, mask=mask)
    tl.store(out10_ptr + offsets, in4_val2, mask=mask)
    
    # Handle tensor split (if this program is also responsible for that)
    total_elements_split = seq_len_half * feat_dim * 2  # Two outputs of equal size
    if pid == 0:  # Let first program handle split operations
        # Second half of the program handles tensor split
        split_offsets = tl.arange(0, total_elements_split)
        split_mask = split_offsets < total_elements_split
        
        # Calculate indices for tensor split
        seq_split = split_offsets // feat_dim
        feat_split = split_offsets % feat_dim
        
        # Load tmp_0 and split along last dimension
        tmp0_idx_first = seq_split * 64 + feat_split  # First half [256, 64]
        tmp0_idx_second = seq_split * 64 + feat_split + 256  # Second half [256, 64] (offset by 256)
        
        # Split the tensor into two parts
        tmp0_val_first = tl.load(tmp0_ptr + tmp0_idx_first, mask=split_mask, other=0.0)
        tmp0_val_second = tl.load(tmp0_ptr + tmp0_idx_second, mask=split_mask, other=0.0)
        
        # Store both split results
        tl.store(out12_ptr + split_offsets, tmp0_val_first, mask=split_mask)
        tl.store(out13_ptr + split_offsets, tmp0_val_second, mask=split_mask)

@torch.fx.wrap  
def optimized_slice_split(in_4, tmp_0):
    batch_size, n_heads, seq_len_full, feat_dim = in_4.shape
    seq_len_half = seq_len_full - 1  # 257 -> 256
    
    # Calculate optimal block size
    total_elements_slice = batch_size * n_heads * seq_len_half * feat_dim
    BLOCK_SIZE = 1024
    num_programs = (total_elements_slice + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    out9 = torch.empty((batch_size, n_heads, seq_len_half, feat_dim), dtype=torch.float32, device=in_4.device)
    out10 = torch.empty((batch_size, n_heads, seq_len_half, feat_dim), dtype=torch.float32, device=in_4.device)
    out12 = torch.empty((tmp_0.shape[0], 64), dtype=torch.float32, device=tmp_0.device)
    out13 = torch.empty((tmp_0.shape[0], 64), dtype=torch.float32, device=tmp_0.device)
    
    # Launch kernel
    slice_split_kernel[(num_programs,)](
        in4_ptr=in_4,
        tmp0_ptr=tmp_0,
        out13_ptr=out13,
        out9_ptr=out9,
        out10_ptr=out10,
        out12_ptr=out12,
        batch_size=batch_size,
        n_heads=n_heads,
        seq_len_full=seq_len_full,
        seq_len_half=seq_len_half,
        feat_dim=feat_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return (out13, out9, out10, out12)

def replacement_func():
    return optimized_slice_split