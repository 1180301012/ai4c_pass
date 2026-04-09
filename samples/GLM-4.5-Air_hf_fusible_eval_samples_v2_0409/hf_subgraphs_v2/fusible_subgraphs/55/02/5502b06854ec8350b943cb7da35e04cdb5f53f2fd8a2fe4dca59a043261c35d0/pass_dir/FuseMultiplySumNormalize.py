import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    return tmp_5

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_multiply_sum_normalize_kernel(
    in_0_ptr,  # int64 tensor, needs conversion
    in_1_ptr,  # float16/bfloat16 tensor
    out_ptr,   # output tensor
    n_batch,   # batch size
    n_seq,     # sequence length  
    n_features, # feature dimension 
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    # Batch index - process one batch at a time
    batch_idx = tl.program_id(0)
    feat_start = tl.program_id(1) * BLOCK_SIZE_FEAT
    
    # Feature offsets
    feat_offset = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
    
    # Create mask for bounds checking
    feat_mask = feat_offset < n_features
    
    # Initialize accumulators - compute per feature position
    sum_prod_all = tl.zeros([BLOCK_SIZE_FEAT], dtype=tl.float32)
    sum_in0_all = tl.zeros([BLOCK_SIZE_FEAT], dtype=tl.float32)
    
    # Process all sequence positions for each feature
    for seq_idx in range(n_seq):
        # Get feature pointers for current sequence position
        in_0_feat_ptr = in_0_ptr + batch_idx * n_seq * n_features + seq_idx * n_features + feat_offset
        in_1_feat_ptr = in_1_ptr + batch_idx * n_seq * n_features + seq_idx * n_features + feat_offset
        
        # Load data with proper type conversion
        in_0_features = tl.load(in_0_feat_ptr, mask=feat_mask, other=0).to(tl.float32)
        in_1_features = tl.load(in_1_feat_ptr, mask=feat_mask, other=0.0)
        
        # Accumulate products and sums vectorized
        sum_prod_all += in_0_features * in_1_features
        sum_in0_all += in_0_features
    
    # Final normalize with clamping - replicate for each sequence position
    clamped_sum_in0 = tl.maximum(sum_in0_all, 1e-09)
    result = sum_prod_all / clamped_sum_in0
    
    # Replicate the result for each sequence position
    for seq_idx in range(n_seq):
        out_idx = batch_idx * n_seq * n_features + seq_idx * n_features + feat_offset
        tl.store(out_ptr + out_idx, result, mask=feat_mask)

@torch.fx.wrap
def fused_multiply_sum_normalize(in_0, in_1):
    # Get tensor shapes
    n_batch, n_seq, n_features = in_0.shape
    
    # Output shape: [batch, seq, features] - replicating pattern from original computation
    out_shape = (n_batch, n_seq, n_features)
    out = torch.empty(out_shape, dtype=torch.float32, device=in_0.device)
    
    # Block size for features
    BLOCK_SIZE_FEAT = 1024  # Process features efficiently
    
    # Calculate grid dimensions - 2D grid (batch, feature blocks)
    grid_x = n_batch
    grid_y = (n_features + BLOCK_SIZE_FEAT - 1) // BLOCK_SIZE_FEAT
    
    # Launch kernel
    fused_multiply_sum_normalize_kernel[(grid_x, grid_y)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_batch=n_batch,
        n_seq=n_seq,
        n_features=n_features,
        BLOCK_SIZE_FEAT=BLOCK_SIZE_FEAT,
    )
    
    return out

def replacement_func():
    return fused_multiply_sum_normalize