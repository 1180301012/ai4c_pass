import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_2):
    tmp_4 = in_2.mean(dim = -2, keepdim = True)
    return tmp_4

# Argument extraction function
def replacement_args(in_2):
    return (in_2,)

@triton.jit
def optimized_mean_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    seq_length,
    feature_dim,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_FEATURE: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_feature = tl.program_id(2)
    
    # Compute offsets
    batch_offset = pid_batch * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    seq_offset = pid_seq * BLOCK_SIZE_SEQ + tl.arange(0, BLOCK_SIZE_SEQ)
    feature_offset = pid_feature * BLOCK_SIZE_FEATURE + tl.arange(0, BLOCK_SIZE_FEATURE)
    
    # Create masks
    batch_mask = batch_offset < n_batch
    seq_mask = seq_offset < seq_length
    feature_mask = feature_offset < feature_dim
    
    # Load input data
    input_ptrs = input_ptr + (
        batch_offset[:, None, None] * (seq_length * feature_dim) +
        seq_offset[None, :, None] * feature_dim +
        feature_offset[None, None, :]
    )
    
    input_data = tl.load(input_ptrs, mask=batch_mask[:, None, None] & seq_mask[None, :, None] & feature_mask[None, None, :], other=0.0)
    
    # Compute mean over sequence dimension (dim=-2, which is seq_length dimension)
    sum_data = tl.sum(input_data, axis=1)
    count_seq = tl.sum(seq_mask.to(tl.int32))
    
    # Normalize by sequence length
    mean_data = sum_data / count_seq.to(tl.float32)
    
    # Store result at corresponding output position
    output_ptrs = output_ptr + (
        batch_offset[:, None] * feature_dim +
        feature_offset[None, :]
    )
    
    output_data = mean_data.reshape(-1, BLOCK_SIZE_FEATURE)
    output_mask = (batch_mask[:, None] & feature_mask[None, :]).reshape(-1)
    
    tl.store(output_ptrs, output_data, mask=output_mask)

@torch.fx.wrap
def optimized_mean_reduction(x):
    # Get input dimensions: [batch, seq_length, feature_dim]
    n_batch, seq_length, feature_dim = x.shape
    
    # Set block sizes for optimal GPU utilization
    BLOCK_SIZE_BATCH = 4   # Reduce batch dimension for better occupancy
    BLOCK_SIZE_SEQ = seq_length  # Process entire sequence in one block
    BLOCK_SIZE_FEATURE = min(256, feature_dim)  # Optimized feature dimension
    
    # Adjust block sizes if needed
    if BLOCK_SIZE_FEATURE > feature_dim:
        BLOCK_SIZE_FEATURE = feature_dim
    
    # Calculate grid size
    num_blocks_batch = (n_batch + BLOCK_SIZE_BATCH - 1) // BLOCK_SIZE_BATCH
    num_blocks_seq = 1  # Process entire sequence
    num_blocks_feature = (feature_dim + BLOCK_SIZE_FEATURE - 1) // BLOCK_SIZE_FEATURE
    
    # Create output tensor with same dtype and device
    # Output shape: [batch, 1, feature_dim] due to keepdim=True
    out = torch.empty((n_batch, 1, feature_dim), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    if n_batch > 0 and feature_dim > 0:
        optimized_mean_kernel[(num_blocks_batch, num_blocks_seq, num_blocks_feature)](
            input_ptr=x,
            output_ptr=out,
            n_batch=n_batch,
            seq_length=seq_length,
            feature_dim=feature_dim,
            BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
            BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
            BLOCK_SIZE_FEATURE=BLOCK_SIZE_FEATURE,
        )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return optimized_mean_reduction