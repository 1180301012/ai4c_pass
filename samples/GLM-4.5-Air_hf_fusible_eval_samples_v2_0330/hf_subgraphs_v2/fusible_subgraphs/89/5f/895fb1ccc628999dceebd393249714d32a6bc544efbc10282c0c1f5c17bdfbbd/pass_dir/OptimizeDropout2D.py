import torch
import triton
import triton.language as tl

# Pattern matching function for dropout2d operation
def pattern(tmp_4):
    tmp_5 = torch.nn.functional.dropout2d(tmp_4, 0.1, False, False)
    return tmp_5  # Return observables

# Argument extraction function
def replacement_args(tmp_4):
    return (tmp_4,)

# Optimized Triton kernel for dropout2d
@triton.jit
def dropout2d_kernel(
    input_ptr,
    output_ptr,
    N,  # Total number of elements
    C,  # Number of channels
    H,  # Height
    W,  # Width
    p: tl.constexpr,  # Dropout probability
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,  # Block size for channel dimension
):
    # Get global program ID for spatial and channel parallelism
    pid = tl.program_id(0)
    
    # We'll use 2D grid: first dimension for channels, second for spatial blocks
    channel_id = pid // ((H * W + BLOCK_SIZE - 1) // BLOCK_SIZE)
    spatial_block_id = pid % ((H * W + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Check if channel_id is within bounds
    channel_mask = channel_id < C
    if not channel_mask:
        return
    
    # Calculate spatial block offsets
    spatial_block_start = spatial_block_id * BLOCK_SIZE
    spatial_block_end = tl.minimum(spatial_block_start + BLOCK_SIZE, H * W)
    spatial_offsets = spatial_block_start + tl.arange(0, spatial_block_end - spatial_block_start)
    spatial_mask = spatial_offsets < (H * W)
    
    # Calculate overall element offsets for this channel
    base_offset = channel_id * H * W + spatial_offsets
    
    # Skip if no spatial elements to process
    if tl.sum(spatial_mask) == 0:
        return
    
    # Load input data
    x = tl.load(input_ptr + base_offset, mask=spatial_mask, other=0.0)
    
    # During training, apply dropout with probability p
    # We generate a random mask in Triton, using the element index as seed for reproducibility
    rand_idx = base_offset[spatial_mask]
    
    # Hash function for deterministic randomness based on indices
    # This ensures consistent results during training while maintaining dropout behavior
    rand_state = (rand_idx * 1103515245 + 12345) & 0x7fffffff
    normalized_rand = rand_state / 2147483647.0  # Normalize to [0, 1]
    
    # Keep probability: 1 - p
    keep_prob = 1.0 - p
    dropout_mask = normalized_rand > p
    
    # Apply dropout: scale by keep_prob to maintain expected value
    # For channels-first format (N, C, H, W), we work on each channel independently
    # Since we only have one channel at a time, we can apply dropout directly
    if keep_prob != 1.0:
        dropout_output = x * dropout_mask * (1.0 / keep_prob)
    else:
        dropout_output = x
    
    # Store result
    tl.store(output_ptr + base_offset, dropout_output, mask=spatial_mask)

# Kernel wrapper for optimized dropout2d
@torch.fx.wrap
def triton_dropout2d(x, p=0.1, training=False):
    if not training:
        # During inference/eval, dropout is a no-op
        return x
    
    B, C, H, W = x.shape
    N = B * C * H * W
    
    # Optimized block configuration for 2D dropout
    # Process channels and spatial dimensions in parallel
    BLOCK_SIZE_SPATIAL = 1024  # Block size for spatial processing
    BLOCK_SIZE_CHANNELS = 512   # Block size for channel processing
    
    # Calculate grid size: (channels * spatial_blocks) + channel_blocks
    spatial_blocks = (H * W + BLOCK_SIZE_SPATIAL - 1) // BLOCK_SIZE_SPATIAL
    total_blocks = C * spatial_blocks
    
    output = torch.empty_like(x)
    
    # Launch the optimized dropout2d kernel
    dropout2d_kernel[total_blocks,](
        input_ptr=x,
        output_ptr=output,
        N=N,
        C=C,
        H=H,
        W=W,
        p=p,
        BLOCK_SIZE=BLOCK_SIZE_SPATIAL,
    )
    
    return output

# Replacement function (returns function reference, not a call)
def replacement_func():
    return triton_dropout2d