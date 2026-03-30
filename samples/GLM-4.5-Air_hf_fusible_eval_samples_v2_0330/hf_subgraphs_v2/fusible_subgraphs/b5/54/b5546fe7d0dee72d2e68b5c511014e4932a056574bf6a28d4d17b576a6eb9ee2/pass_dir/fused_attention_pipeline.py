import torch
import triton
import triton.language as tl

def pattern(in_2, in_3, conv2d, in_6, in_4, scalar_mult):
    """
    Pattern matching for the attention pipeline computation:
    cat + reshape + transpose + multiply + pad + scalar_mult + add + transpose + reshape
    """
    tmp_3 = torch.cat([in_2, in_3, conv2d], dim=1)
    tmp_4 = tmp_3.reshape(1, 8, -1, tmp_3.shape[-1] // (in_2.shape[0] * 8))
    tmp_5 = tmp_4.transpose(-1, -2)
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = scalar_mult * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, tmp_10.shape[2], -1)
    return tmp_11

def replacement_args(in_2, in_3, conv2d, in_6, in_4, scalar_mult):
    """Extract arguments for the fused kernel"""
    return (in_2, in_3, conv2d, in_6, in_4, scalar_mult)

@triton.jit
def fused_attention_kernel(
    in2_ptr, in3_ptr, conv2d_ptr, in6_ptr, in4_ptr, out_ptr,
    in2_shape, in3_shape, conv2d_shape, in6_shape, in4_shape,
    batch, channels, height1, width1, height2, width2, 
    scalar_val: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """Optimized Triton kernel for fused attention pipeline"""
    pid = tl.program_id(0)
    
    # Calculate spatial dimensions after reshape/concat
    concat_channels = in2_shape[1] + in3_shape[1] + conv2d_shape[1]
    feature_dim = concat_channels // 8
    
    # Generate offsets
    m_offs = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offs = tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    m_mask = m_offs < batch * feature_dim
    n_mask = n_offs < width2
    
    # Load data efficiently with batch processing
    # Load concatenated channels
    in2_ptrs = in2_ptr + m_offs.reshape(-1, 1) * in2_shape[1] * in2_shape[2] + n_offs
    in3_ptrs = in3_ptr + m_offs.reshape(-1, 1) * in3_shape[1] * in3_shape[2] + n_offs
    conv2d_ptrs = conv2d_ptr + m_offs.reshape(-1, 1) * conv2d_shape[1] * conv2d_shape[2] + n_offs
    
    in2_data = tl.load(in2_ptrs, mask=(m_mask.reshape(-1, 1) & n_mask.reshape(1, -1)), other=0.0)
    in3_data = tl.load(in3_ptrs, mask=(m_mask.reshape(-1, 1) & n_mask.reshape(1, -1)), other=0.0)
    conv2d_data = tl.load(conv2d_ptrs, mask=(m_mask.reshape(-1, 1) & n_mask.reshape(1, -1)), other=0.0)
    
    # Concatenate channels
    concat_data = tl.concatenate([in2_data, in3_data, conv2d_data], dim=1)
    
    # Reshape and transpose: [batch, 8, feature_dim, width2] -> [batch, feature_dim, 8, width2] -> [batch, feature_dim, width2*8]
    concat_data = concat_data.reshape(BLOCK_SIZE_M, 8, feature_dim, BLOCK_SIZE_N)
    transposed = concat_data.transpose(0, 2).reshape(BLOCK_SIZE_M * feature_dim, 8 * BLOCK_SIZE_N)
    
    # Load multiplier tensor
    in6_ptrs = in6_ptr + m_offs.reshape(-1, 1) * in6_shape[2] * in6_shape[3] + n_offs.reshape(1, -1)
    in6_data = tl.load(in6_ptrs, mask=(m_mask.reshape(-1, 1) & (n_offs < in6_shape[3])).reshape(1, -1), other=0.0)
    
    # Element-wise multiplication
    mult_result = transposed.to(tl.float32) * in6_data.to(tl.float32)
    
    # Apply padding (add row at the beginning) - simulated by shifting
    pad_offset = (m_offs < batch * feature_dim - 1).to(tl.int32)
    padded_result = tl.where(pad_offset.reshape(-1, 1), mult_result, 0.0)
    
    # Load scalar multiplier tensor and apply scaling
    in4_ptrs = in4_ptr + m_offs.reshape(-1, 1) * in4_shape[2] * in4_shape[3] + n_offs.reshape(1, -1)
    in4_data = tl.load(in4_ptrs, mask=(m_mask.reshape(-1, 1) & (n_offs < in4_shape[3])).reshape(1, -1), other=0.0)
    
    scaled_in4 = in4_data.to(tl.float32) * scalar_val
    
    # Add results (broadcasting properly)
    final_result = padded_result + scaled_in4
    
    # Final transpose and reshape to output format
    # [batch*feature_dim, width2*8] -> [batch, feature_dim, width2*8] -> [batch, feature_dim*8, width2] -> transpose -> [batch, width2, feature_dim*8]
    final_result_reshaped = final_result.reshape(batch, feature_dim, 8 * BLOCK_SIZE_N)
    output_result = final_result_reshaped.transpose(1, 2).reshape(batch * BLOCK_SIZE_N, feature_dim * 8)
    
    # Store result
    out_ptrs = out_ptr + m_offs.reshape(-1, 1) * batch * feature_dim * 8 + n_offs.reshape(1, -1)
    tl.store(out_ptrs, output_result.to(out_ptr.dtype.element_type), 
             mask=(m_mask.reshape(-1, 1) & (n_offs < batch * feature_dim * 8)).reshape(1, -1))

@torch.fx.wrap
def fused_attention_pipeline(in_2, in_3, conv2d, in_6, in_4, scalar_mult):
    """Wrapper function for fused attention pipeline"""
    batch = 1  # All models have batch=1
    in2_shape = in_2.shape
    in3_shape = in_3.shape
    conv2d_shape = conv2d.shape
    in6_shape = in_6.shape
    in4_shape = in_4.shape
    
    # Calculate feature dimensions
    concat_channels = in2_shape[1] + in3_shape[1] + conv2d_shape[1]
    feature_dim = concat_channels // 8
    
    # Determine output dimensions based on pattern analysis
    if in6_shape[2] * in6_shape[3] > 1000:  # Large spatial dimensions
        final_height = in6_shape[2] + 1  # Account for padding
        final_width = concat_channels // 8
    else:
        final_height = in6_shape[2] + 1
        final_width = concat_channels // 8
    
    # Output tensor
    output_shape = (1, final_height, final_width * 8)
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Set block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 128
    
    # Calculate grid size
    grid_size = (batch * feature_dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch kernel
    fused_attention_kernel[grid_size, (
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )](
        in_2, in_3, conv2d, in_6, in_4, output,
        in2_shape, in3_shape, conv2d_shape, in6_shape, in4_shape,
        batch, in2_shape[1] + in3_shape[1] + conv2d_shape[1], 
        feature_dim, in6_shape[2], in6_shape[3],
        scalar_mult, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    """Return the fused attention pipeline function"""
    return fused_attention_pipeline