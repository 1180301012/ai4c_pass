import torch
import triton
import triton.language as tl


@triton.jit
def fused_attention_kernel(
    # Input pointers
    in_0_ptr,      # [1, 2, 256, H, W] - feature maps
    in_1_ptr,      # [1, 2, 256, 1, 1] - attention scores
    out_ptr,       # [1, 256, H, W] - output
    
    # Shape information (all compile-time constants)
    n_channels,    # 256 (number of channels)
    n_heads,       # 2 (number of attention heads)
    height,        # 32 or 8 (spatial height)
    width,         # 32 or 8 (spatial width)
    
    # Constants
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs automatically from Triton
    channel_block_id = tl.program_id(0)
    spatial_block_id = tl.program_id(1)
    
    # Calculate offsets for this block
    channel_offset = channel_block_id * BLOCK_SIZE
    spatial_offset = spatial_block_id * BLOCK_SIZE
    
    # Create indices within this block
    channel_idx = tl.arange(0, BLOCK_SIZE)
    spatial_idx = tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid indices
    channel_mask = channel_offset + channel_idx < n_channels
    spatial_mask = spatial_offset + spatial_idx < (height * width)
    
    # Attention scores are computed per channel
    attn_scores = tl.load(in_1_ptr + channel_offset + channel_idx, mask=channel_mask, other=0.0)
    
    # Compute softmax attention scores for each head separately
    # Since we have 2 heads, handle them explicitly for better performance
    softmax_head0 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    softmax_head1 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    if n_heads >= 2:
        # Load head scores separately (assuming they're stored sequentially)
        head0_scores = attn_scores  # First head
        head1_scores = tl.load(in_1_ptr + channel_offset + channel_idx + n_channels, 
                              mask=channel_mask, other=0.0)  # Second head
        
        # Compute softmax
        max_scores = tl.maximum(head0_scores, head1_scores)
        exp0 = tl.exp(head0_scores - max_scores)
        exp1 = tl.exp(head1_scores - max_scores)
        sum_exp = exp0 + exp1
        softmax_head0 = exp0 / sum_exp
        softmax_head1 = exp1 / sum_exp
    
    # Initialize output accumulator
    output_channels = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process each head
    for head in range(n_heads):
        # Calculate feature base offset for this head
        head_feat_offset = head * n_channels * height * width
        
        # Load features for each spatial position in this block
        feat_ptr = head_feat_offset + (channel_offset + channel_idx) * height * width + (spatial_offset + spatial_idx)
        # Use a simpler mask approach - just check if each channel is valid
        features = tl.load(in_0_ptr + feat_ptr, mask=channel_mask, other=0.0)
        
        # Get attention scores for this head
        if head == 0:
            head_attn = softmax_head0
        else:
            head_attn = softmax_head1
        
        # Apply attention and accumulate
        weighted_features = features * head_attn
        output_channels += weighted_features
    
    # Store result
    out_base_offset = channel_offset * height * width
    out_offset = out_base_offset + (spatial_offset + spatial_idx)
    
    tl.store(out_ptr + out_offset, output_channels, mask=spatial_mask)


@torch.fx.wrap
def fused_attention_kernel_wrapper(in_0, in_1):
    # Get input shapes
    shape_0 = in_0.shape
    shape_1 = in_1.shape
    
    n_batch, n_heads, n_channels, height, width = shape_0
    _, _, _, _, _ = shape_1
    
    # Calculate output shape
    out_shape = (n_batch, n_channels, height, width)
    out = torch.zeros(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Use even smaller block size for better performance
    # Try 32 for potentially better cache utilization and lower overhead
    BLOCK_SIZE = 32
    
    # Calculate grid dimensions
    spatial_size = height * width
    channel_blocks = (n_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    spatial_blocks = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with optimized grid
    fused_attention_kernel[(channel_blocks, spatial_blocks)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_channels=n_channels,
        n_heads=n_heads,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    """Match softmax -> multiply -> sum pattern"""
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2


def replacement_args(in_0, in_1):
    """Extract arguments for the replacement function"""
    return (in_0, in_1)


def replacement_func():
    """Return the fused attention kernel wrapper"""
    return fused_attention_kernel_wrapper