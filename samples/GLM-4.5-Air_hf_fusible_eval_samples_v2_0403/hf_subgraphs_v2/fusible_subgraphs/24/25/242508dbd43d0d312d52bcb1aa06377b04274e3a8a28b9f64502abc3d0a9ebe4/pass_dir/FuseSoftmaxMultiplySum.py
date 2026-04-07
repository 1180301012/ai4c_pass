import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """
    Pattern matching for: softmax -> multiply -> sum
    This matches the computation pattern found in the model:
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0  
    tmp_2 = torch.sum(tmp_1, dim=1)
    """
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel with performance optimizations
@triton.jit
def fused_attention_kernel(
    # Input pointers
    in_0_ptr,
    in_1_ptr,
    # Output pointer  
    out_ptr,
    # Tensor shapes
    batch_size,
    num_channels,
    height,
    width,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program IDs - batch and spatial program ID
    batch_id = tl.program_id(0)
    spatial_program_id = tl.program_id(1)
    
    # Calculate total number of channel-spatial elements to process
    total_elements = num_channels * height * width
    spatial_elements_per_channel = height * width
    
    # Process spatial elements within this channel with optimized vectorization
    offsets = tl.arange(0, BLOCK_SIZE)
    spatial_offsets = spatial_program_id * BLOCK_SIZE + offsets
    mask = spatial_offsets < total_elements
    
    # Decode spatial offsets to channel coordinates (simplified)
    channel_coords = spatial_offsets // spatial_elements_per_channel
    
    # Load attention scores for both elements of the 2-element vector
    in_1_base = in_1_ptr + batch_id * num_channels * 2 + channel_coords * 2
    score0 = tl.load(in_1_base, mask=channel_coords < num_channels, other=0.0)
    score1 = tl.load(in_1_base + 1, mask=channel_coords < num_channels, other=0.0)
    
    # Optimized 2-element softmax computation
    score0_fp32 = score0.to(tl.float32)
    score1_fp32 = score1.to(tl.float32)
    max_val = tl.maximum(score0_fp32, score1_fp32)
    
    # Combined exp and division for better efficiency
    exp0 = tl.exp(score0_fp32 - max_val)
    exp1 = tl.exp(score1_fp32 - max_val)
    sum_exp = exp0 + exp1
    softmax_weight = exp1 / sum_exp
    
    # Load feature map with optimized memory access pattern
    in_0_offset = batch_id * num_channels * spatial_elements_per_channel + spatial_offsets
    features = tl.load(in_0_ptr + in_0_offset, mask=mask, other=0.0).to(softmax_weight.dtype)
    
    # Final multiplication with fused operation
    result = features * softmax_weight
    
    # Store result with optimal memory coalescing
    out_offset = batch_id * num_channels * spatial_elements_per_channel + spatial_offsets
    tl.store(out_ptr + out_offset, result, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_attention_kernel_wrapper(in_0, in_1):
    batch_size, num_channels, height, width = in_0.shape[0], in_0.shape[2], in_0.shape[3], in_0.shape[4]
    
    # Create output tensor - sum over the 2-element dimension (dim=1)
    out = torch.empty((batch_size, 1, num_channels, height, width), dtype=in_0.dtype, device=in_0.device)
    
    # Total elements across channels and spatial dimensions
    total_elements = num_channels * height * width
    
    # Configure optimal block size with power of 2 constraint
    if total_elements <= 0:
        BLOCK_SIZE = 1
    else:
        # Optimal block size for this workload type
        BLOCK_SIZE = min(256, total_elements)
        # Ensure it's a power of 2 for triton compatibility
        BLOCK_SIZE = 1 if BLOCK_SIZE <= 0 else 2 ** ((BLOCK_SIZE - 1).bit_length())
        BLOCK_SIZE = min(BLOCK_SIZE, total_elements)
    
    # Calculate grid dimensions (2D grid: batch and flattened channel-spatial elements)  
    num_spatial_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch_size, num_spatial_blocks)
    
    # Launch kernel
    fused_attention_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_attention_kernel_wrapper