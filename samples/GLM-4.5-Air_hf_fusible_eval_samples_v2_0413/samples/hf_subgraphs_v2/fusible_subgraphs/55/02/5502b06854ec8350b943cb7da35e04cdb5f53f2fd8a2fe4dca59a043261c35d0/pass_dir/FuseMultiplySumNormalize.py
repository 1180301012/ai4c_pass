import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    # Pattern matches the entire computation chain except the final redundant concat
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    return tmp_5

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel
@triton.jit
def fused_multiply_sum_normalize_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_batch,
    n_seq,
    n_feat,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature dimension
    b = tl.program_id(0)  # batch
    f = tl.program_id(1)  # feature
    
    # Initialize accumulators for this feature across all sequences
    weighted_sum_all_seq = 0.0
    norm_sum_all_seq = 0.0
    
    # For single feature processing (f is fixed), load and sum across all sequences
    for s in range(n_seq):
        # Calculate offset for this batch, sequence, and specific feature
        offset = b * n_seq * n_feat + s * n_feat + f
        
        # Load weight (convert to float32) and feature values
        weight = tl.load(in_0_ptr + offset).to(tl.float32)
        feature = tl.load(in_1_ptr + offset)
        
        # Accumulate weighted sum and normalization factor
        weighted_sum_all_seq += weight * feature
        norm_sum_all_seq += weight
    
    # Apply clamping and normalization (clamping is redundant for single values but included for correctness)
    clamped_norm = tl.maximum(norm_sum_all_seq, 1e-09)
    result = weighted_sum_all_seq / clamped_norm
    
    # Store result for this feature
    out_offset = b * n_feat + f
    tl.store(out_ptr + out_offset, result)

@torch.fx.wrap
def fused_multiply_sum_normalize(in_0, in_1):
    # Input shapes: [1, 10, 1024]
    batch_size = in_0.size(0)
    seq_len = in_0.size(1)
    feat_dim = in_0.size(2)
    
    # Create output tensor using allowed operations
    output = torch.empty(batch_size, feat_dim, dtype=torch.float32, device=in_0.device)
    
    # Use smaller block size for better occupancy with small feature dimension
    BLOCK_SIZE = 256
    n_programs = (feat_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_multiply_sum_normalize_kernel[(batch_size, n_programs)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=output,
        n_batch=batch_size,
        n_seq=seq_len,
        n_feat=feat_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return fused_multiply_sum_normalize