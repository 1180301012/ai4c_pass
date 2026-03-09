# Deprecated pass - replaced by FuseOptimizedWeightedSum

@triton.jit
def fused_softmax_weighted_sum_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    batch_size,
    softmax_channels,      # This should be 2 (dim=1)
    softmax_seq_len,       # This should be 128 (last dim)
    in_0_channels,         # This should be 128 (dim=2)
    spatial_size_0,        # Spatial dimension 1 (48 or 64)
    spatial_size_1,        # Spatial dimension 2 (64 or 48)
    BLOCK_SOFTMAX: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    # Program identifiers
    batch_pid = tl.program_id(0)
    softmax_pid = tl.program_id(1)
    spatial_0_pid = tl.program_id(2)
    spatial_1_pid = tl.program_id(3)
    
    # Calculate offsets for each dimension
    batch_offset = batch_pid
    softmax_offset = softmax_pid * BLOCK_SOFTMAX
    spatial_0_offset = spatial_0_pid * BLOCK_SPATIAL
    spatial_1_offset = spatial_1_pid * BLOCK_SPATIAL
    
    # Softmax computation
    softmax_start_ptr = in_1_ptr + batch_offset * (softmax_channels * softmax_seq_len) + softmax_offset * softmax_channels
    
    # Load softmax input elements for the current block
    softmax_row_offsets = tl.arange(0, softmax_channels)
    softmax_col_offsets = tl.arange(0, BLOCK_SOFTMAX)
    softmax_offsets = softmax_row_offsets[:, None] * softmax_seq_len + softmax_col_offsets[None, :]
    softmax_mask = (softmax_offsets < softmax_seq_len) & (softmax_row_offsets[:, None] < softmax_channels)
    
    softmax_input = tl.load(softmax_start_ptr + softmax_offsets, mask=softmax_mask, other=-float('inf'))
    
    # Compute softmax (max for numerical stability)
    max_val = tl.max(softmax_input, axis=1)
    softmax_exp = tl.exp(softmax_input - max_val[:, None])
    softmax_sum = tl.sum(softmax_exp, axis=1)
    softmax_output = softmax_exp / softmax_sum[:, None]
    
    # Reshape softmax output to match: [batch, channels=2, seq_len=128, 1, 1]
 # We need to expand the softmax output to multiply with in_0
    # For simplicity, we'll do a simplified version that assumes we can broadcast
    # Load in_0 elements for the current block
    in_0_ptr_offset = (batch_offset * in_0_channels * spatial_size_0 * spatial_size_1 +
                      softmax_offset * spatial_size_0 * spatial_size_1 +
                      spatial_0_offset * spatial_size_1 + spatial_1_offset)
    
    in_0_offsets = tl.arange(0, BLOCK_SPATIAL)
    in_0_mask = in_0_offsets < min(BLOCK_SPATIAL, spatial_size_1 - spatial_1_offset)
    
    # Load spatial elements
    for i in range(BLOCK_SOFTMAX):
        if softmax_offset + i < softmax_seq_len:
            softmax_flat_idx = i * softmax_channels  # Flatten to get the right channel
            softmax_weight = tl.load(softmax_start_ptr + softmax_flat_idx, mask=softmax_flat_idx < softmax_channels, other=0.0)
            
            # Load in_0 elements for current softmax channel and spatial position
            in_0_elem = tl.load(in_0_ptr + in_0_ptr_offset, mask=in_0_mask, other=0.0)
            
            # Weighted multiplication (simplified - assumes broadcasting)
            weighted_sum = softmax_weight * in_0_elem
            
            if batch_pid == 0 and softmax_pid == 0 and spatial_0_pid == 0 and spatial_1_pid == 0:
                # Store to output
                out_offset = (batch_pid * in_0_channels * spatial_size_0 * spatial_size_1 +
                             i * spatial_size_0 * spatial_size_1 +
                             spatial_0_pid * spatial_size_1 + spatial_1_pid)
                tl.store(out_ptr + out_offset, weighted_sum, mask=in_0_mask)

@torch.fx.wrap
def fused_softmax_weighted_sum(in_0, in_1):
    # Get tensor shapes
    batch_size = in_0.shape[0]
    spatial_size_0 = in_0.shape[3]
    spatial_size_1 = in_0.shape[4]
    in_0_channels = in_0.shape[2]
    softmax_channels = in_1.shape[1]
    softmax_seq_len = in_1.shape[3]
    
    # Output shape after sum along dim=1: [batch, in_0_channels, spatial_size_0, spatial_size_1]
    output_shape = (batch_size, in_0_channels, spatial_size_0, spatial_size_1)
    output = torch.empty(output_shape, dtype=torch.float32, device=in_0.device)
    
    # Block sizes for GPU optimization
    BLOCK_SOFTMAX = 128  # Process full softmax sequence length
    BLOCK_SPATIAL = 256   # Process spatial blocks efficiently
    
    # Calculate grid size
    softmax_blocks = (softmax_seq_len + BLOCK_SOFTMAX - 1) // BLOCK_SOFTMAX
    spatial_0_blocks = (spatial_size_0 + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL
    spatial_1_blocks = (spatial_size_1 + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL
    
    grid = (batch_size, softmax_blocks, spatial_0_blocks, spatial_1_blocks)
    
    # Launch the kernel
    fused_softmax_weighted_sum_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=output,
        batch_size=batch_size,
        softmax_channels=softmax_channels,
        softmax_seq_len=softmax_seq_len,
        in_0_channels=in_0_channels,
        spatial_size_0=spatial_size_0,
        spatial_size_1=spatial_size_1,
        BLOCK_SOFTMAX=BLOCK_SOFTMAX,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
    )
    
    return output

def replacement_func():
    return fused_softmax_weighted_sum