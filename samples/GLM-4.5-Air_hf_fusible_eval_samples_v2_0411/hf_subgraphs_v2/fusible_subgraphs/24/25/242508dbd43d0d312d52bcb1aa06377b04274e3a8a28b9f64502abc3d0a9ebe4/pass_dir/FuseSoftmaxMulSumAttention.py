import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(in_0, in_1):
    """Match the computation pattern: softmax + multiplication + sum"""
    tmp_0 = torch.softmax(in_1, dim = 1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim = 1)
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Helper kernel for computing softmax across the 2-element dimension
@triton.jit
def softmax_kernel_2dim(
    in_1_ptr,  # [B, 2, C, 1, 1] - original shape
    softmax_ptr,  # [B, 2, C] - output shape
    batch_size,
    channels,
    BLOCK_SIZE_CHANNELS: tl.constexpr,
):
    """Compute softmax across the 2-element dimension for each batch and channel"""
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1) * BLOCK_SIZE_CHANNELS + tl.arange(0, BLOCK_SIZE_CHANNELS)
    channel_mask = channel_idx < channels
    
    if batch_idx < batch_size:
        # Load weights for this batch and channel [2, C, 1, 1] -> [2]
        in_1_base = in_1_ptr + (
            batch_idx * 2 * channels * 1 * 1 +  
            channel_idx * 1 * 1
        )
        weights = tl.load(in_1_base, mask=channel_mask, other=0.0)
        
        # Compute softmax across the 2-element dimension
        max_val = tl.max(weights)
        exp_weights = tl.exp(weights - max_val)
        sum_exp = tl.sum(exp_weights)
        softmax_weights = exp_weights / sum_exp
        
        # Store softmax results [B, 2, C]
        softmax_base = softmax_ptr + (batch_idx * 2 * channels + channel_idx)
        tl.store(softmax_base, softmax_weights, mask=channel_mask)

# Optimized Triton kernel for fused attention computation
@triton.jit
def fused_attention_kernel(
    # Input tensors
    in_0_ptr,  # [B, 2, C, H, W] - spatial features
    softmax_ptr,  # [B, 2, C] - precomputed softmax weights
    # Output tensor
    out_ptr,   # [B, C, H, W] - final summed features
    # Metadata
    batch_size,
    channels,
    height,
    width,
    # Data type
    element_size,
    # Triton constants
    BLOCK_SIZE_CHANNELS: tl.constexpr,
):
    """Fused kernel: multiply by softmax weights and sum across dim=1"""
    
    # Batch and channel indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1) * BLOCK_SIZE_CHANNELS + tl.arange(0, BLOCK_SIZE_CHANNELS)
    channel_mask = channel_idx < channels
    
    # Process each batch and channel
    if batch_idx < batch_size:
        # Process one channel at a time
        # Load softmax weights for this channel
        softmax_base = softmax_ptr + (batch_idx * 2 * channels + channel_idx)
        w0 = tl.load(softmax_base + 0, mask=channel_mask, other=0.0)
        w1 = tl.load(softmax_base + 1, mask=channel_mask, other=0.0)
        
        # Process spatial locations: [H x W]
        for spatial_idx in range(height * width):
            # Load and combine features from both dim2 elements
            # Calculate proper memory offsets for tensor shape [B, 2, C, H, W]
            spatial_total = height * width
            
            # Load element 0 at this batch, channel, spatial position
            offset0 = (
                batch_idx * 2 * channels * spatial_total +
                0 * channels * spatial_total +
                channel_idx * spatial_total +
                spatial_idx
            ) * element_size
            f0 = tl.load(in_0_ptr + offset0, mask=channel_mask, other=0.0)
            
            # Load element 1 at this batch, channel, spatial position  
            offset1 = (
                batch_idx * 2 * channels * spatial_total +
                1 * channels * spatial_total +
                channel_idx * spatial_total +
                spatial_idx
            ) * element_size
            f1 = tl.load(in_0_ptr + offset1, mask=channel_mask, other=0.0)
            
            # Apply softmax weights and sum
            # Use simple multiplication for now
            result = f0 * w0 + f1 * w1
            
            # Store result
            out_offset = out_ptr + (
                batch_idx * channels * spatial_total +
                channel_idx * spatial_total +
                spatial_idx
            ) * element_size
            tl.store(out_offset, result, mask=channel_mask)

# Kernel wrapper for fused attention computation
@torch.fx.wrap
def fused_attention_wrapper(in_0, in_1):
    """Wrapper function for the fused attention kernel using only Triton operations"""
    B, dim2_size, C, H, W = in_0.shape  # [B, 2, C, H, W]
    
    # Create output tensor preserving input data type
    out = torch.empty((B, C, H, W), dtype=in_0.dtype, device=in_0.device)
    
    # Create intermediate buffer for softmax results [B, 2, C]
    # Use float32 for softmax for numerical stability
    softmax_ptr = torch.empty((B, 2, C), dtype=torch.float32, device=in_0.device)
    
    # Extract metadata from input tensors
    in_0_element_size = in_0.element_size()
    in_1_element_size = in_1.element_size()
    
    # Set up kernel configuration
    BLOCK_SIZE_CHANNELS = 64  # Number of channels processed per thread block
    out_element_size = out.element_size()  # Get output tensor element size
    
    # Calculate grid sizes
    batch_grid = B
    channel_grid = (C + BLOCK_SIZE_CHANNELS - 1) // BLOCK_SIZE_CHANNELS
    
    # Launch softmax computation kernel first using original tensor shapes
    # Tensor shapes: in_1 [B, 2, C, 1, 1], softmax_ptr [B, 2, C]
    softmax_grid = (batch_grid, channel_grid)
    softmax_kernel_2dim[softmax_grid](
        in_1, softmax_ptr.reshape(-1), B, C, BLOCK_SIZE_CHANNELS
    )
    
    # Launch fused attention kernel 
    fusion_grid = (batch_grid, channel_grid)
    fused_attention_kernel[fusion_grid](
        in_0, softmax_ptr.reshape(-1), out, B, C, H, W, out_element_size,
        BLOCK_SIZE_CHANNELS
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_attention_wrapper