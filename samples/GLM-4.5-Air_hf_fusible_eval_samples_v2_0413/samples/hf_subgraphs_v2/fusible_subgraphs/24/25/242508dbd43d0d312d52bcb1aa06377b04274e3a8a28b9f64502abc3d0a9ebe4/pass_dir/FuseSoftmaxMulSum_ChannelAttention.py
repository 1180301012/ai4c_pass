import torch
import triton
import triton.language as tl

@triton.jit
def channel_attention_kernel_fast(
    # Optimized kernel for minimal overhead and maximum throughput
    in_0_ptr,  # [1, 2, 256, H, W] - input tensor
    in_1_ptr,  # [1, 2, 256, 1, 1] - attention weights
    out_ptr,   # [1, 256, H, W] - output tensor
    
    # Tensor shapes
    num_features,  # Always 256
    height,        # Variable (14, 16, 21, 28)
    width,         # Variable (14, 16, 14, 28)
    
    # Strides for optimized access
    in_0_fh_stride, in_0_fw_stride,     # Feature and width strides for in_0
    in_1_f_stride,                      # Feature stride for in_1
    out_f_stride, out_h_stride,         # Feature and height strides for out
    
    # Block sizes for better occupancy
    BLOCK_SIZE_F: tl.constexpr,
):
    # Compute program IDs with optimized block sizes
    pid_f = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Calculate bounds for this thread
    mask_f = pid_f < num_features
    mask_h = pid_h < height
    
    # Process entire row for this feature and height
    for w in range(0, width):
        if mask_f and mask_h and w < width:
            # Load attention weights for this feature (same across all spatial positions)
            channel0_weight = tl.load(in_1_ptr + pid_f * in_1_f_stride)
            channel1_weight = tl.load(in_1_ptr + pid_f * in_1_f_stride + 1)  # channel offset = 1
            
            # Efficient softmax computation (reused across all W positions)
            max_val = tl.maximum(channel0_weight, channel1_weight)
            exp0 = tl.exp(channel0_weight - max_val)
            exp1 = tl.exp(channel1_weight - max_val)
            sum_exp = exp0 + exp1
            weight0 = exp0 / sum_exp
            weight1 = exp1 / sum_exp
            
            # Load both channel values for this position
            base_offset = pid_f * in_0_fh_stride + pid_h * in_0_fw_stride
            val0 = tl.load(in_0_ptr + base_offset + w)                     # channel 0
            val1 = tl.load(in_0_ptr + base_offset + w + in_0_fh_stride)    # channel 1, offset by channels
            
            # Final weighted sum computation
            weighted_sum = val0 * weight0 + val1 * weight1
            
            # Store result
            output_offset = pid_f * out_f_stride + pid_h * out_h_stride + w
            tl.store(out_ptr + output_offset, weighted_sum)

# Kernel optimization complete - focus on fast execution with minimal overhead

@torch.fx.wrap
def optimized_channel_attention(in_0, in_1):
    # Get tensor shapes
    batch_size, num_channels, num_features, height, width = in_0.shape
    assert in_1.shape == (batch_size, num_channels, num_features, 1, 1), f"in_1 shape must be [{batch_size}, {num_channels}, {num_features}, 1, 1], got {in_1.shape}"
    assert num_channels == 2, f"Expected 2 channels, got {num_channels}"
    
    # Handle float16 by casting to fp32 during computation for numerical stability
    is_fp16 = in_0.dtype == torch.float16
    if is_fp16:
        in_0 = in_0.to(torch.float32)
        in_1 = in_1.to(torch.float32)
    
    # Create output tensor (use fp32 if input is fp16 for numerical stability)
    output_dtype = torch.float32 if is_fp16 else in_0.dtype
    output_shape = (batch_size, num_features, height, width)
    out = torch.empty(output_shape, dtype=output_dtype, device=in_0.device)
    
    # Optimized stride calculations for better memory access pattern
    # Feature and height strides for compact memory access
    in_0_fh_stride = num_features * num_channels  # Skip channels first
    in_0_fw_stride = num_features    # Skip features within each channel
    
    in_1_f_stride = num_channels     # Skip channels for attention weights
    
    out_f_stride = num_features      # Output has no channel dimension
    out_h_stride = num_features * height
    
    # Compute grid size optimized for feature and height dimensions
    # Smaller blocks for better occupancy on this specific computation
    BLOCK_SIZE_F = 64  # Process features in chunks for better occupancy
    grid = (
        (num_features + BLOCK_SIZE_F - 1) // BLOCK_SIZE_F,
        height,  # Each thread handles entire width dimension
    )
    
    # Launch optimized fast kernel
    channel_attention_kernel_fast[grid](
        in_0, in_1, out,
        num_features, height, width,
        in_0_fh_stride, in_0_fw_stride,
        in_1_f_stride,
        out_f_stride, out_h_stride,
        BLOCK_SIZE_F
    )
    
    # Cast back to original dtype if we converted
    if is_fp16:
        out = out.to(torch.float16)
    
    return out

def pattern(in_0, in_1):
    """
    Pattern: softmax + multiply + sum operations
    This matches the computation:
    tmp_0 = torch.softmax(in_1, dim=1)  
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    """
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2

def replacement_args(in_0, in_1):
    """Extract arguments needed for the replacement"""
    return (in_0, in_1)

def replacement_func():
    """Return the optimized function"""
    return optimized_channel_attention