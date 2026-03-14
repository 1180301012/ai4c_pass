import torch
import triton
import triton.language as tl

def pattern(in_2, in_3):
    # Simple addition fusion
    tmp_2 = in_3 + in_2
    return tmp_2

def replacement_args(in_2, in_3):
    return (in_2, in_3)

@triton.jit
def fused_add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_batch, n_seq, n_features,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Each program handles a block of features for one batch and sequence pair
    batch_idx = tl.program_id(0) // n_seq
    seq_idx = tl.program_id(0) % n_seq
    feat_start = tl.program_id(1) * BLOCK_SIZE_M
    feat_offsets = feat_start + tl.arange(0, BLOCK_SIZE_M)
    
    mask = feat_offsets < n_features
    
    # Load inputs for this block
    offsets = batch_idx * n_seq * n_features + seq_idx * n_features + feat_offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    sum_val = x + y
    
    # Store result
    tl.store(out_ptr + offsets, sum_val, mask=mask)

@torch.fx.wrap
def fused_add(in_2, in_3):
    """Fused addition operation using Triton with optimized kernel launch"""
    x, y = in_2, in_3
    
    n_batch, n_seq, n_features = x.shape
    
    # Use optimal block size for features
    BLOCK_SIZE_M = min(256, n_features)  # Process features in blocks of 256
    
    # Calculate number of blocks needed
    n_batch_seq = n_batch * n_seq
    n_feature_blocks = (n_features + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Configure grid: [batch_seq_pairs, feature_blocks]
    grid = (
        n_batch_seq,
        n_feature_blocks
    )
    
    # Perform fused addition using Triton kernel
    out = torch.empty_like(x)
    fused_add_kernel[grid](
        x, y, out,
        n_batch, n_seq, n_features,
        BLOCK_SIZE_M
    )
    
    return out

def replacement_func():
    return fused_add