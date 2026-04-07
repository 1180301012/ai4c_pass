import torch
import triton
import triton.language as tl

# Pattern matching for scale + add + batch_norm fusion
def pattern(conv_out, scale_factor, residual, bn_mean, bn_var, bn_weight, bn_bias):
    """
    Match the pattern: scaling + addition + batch_norm
    This is the fused pattern after eliminating the no-op dropout
    """
    # Apply scaling
    scaled = conv_out * scale_factor
    
    # Add residual connection  
    added = residual + scaled
    
    # Apply batch normalization (with identity parameters)
    # batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    normalized = torch.nn.functional.batch_norm(added, bn_mean, bn_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    
    return normalized, added  # Return both outputs as in original

# Argument extraction function
def replacement_args(conv_out, scale_factor, residual, bn_mean, bn_var, bn_weight, bn_bias):
    return (conv_out, scale_factor, residual, bn_mean, bn_var, bn_weight, bn_bias)

# Optimized fused kernel
@triton.jit
def fused_scale_add_batch_norm_kernel(
    conv_out_ptr,
    scale_factor_ptr,
    residual_ptr,
    bn_mean_ptr,
    bn_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    normalized_out_ptr,
    added_out_ptr,
    n_channels,
    height,
    width,
    batch_size,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Calculate program indices
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    h_block_id = tl.program_id(2)
    w_block_id = tl.program_id(3)
    
    # Calculate memory offsets
    batch_offset = batch_id * n_channels * height * width
    channel_offset = channel_id * height * width
    h_offset = h_block_id * BLOCK_SIZE_H
    w_offset = w_block_id * BLOCK_SIZE_W
    
    # Load scale factor (broadcasted across spatial dimensions)
    scale_factor = tl.load(scale_factor_ptr + channel_id)
    
    # Load batch norm parameters for this channel
    mean = tl.load(bn_mean_ptr + channel_id)
    var = tl.load(bn_var_ptr + channel_id)
    weight = tl.load(bn_weight_ptr + channel_id)
    bias = tl.load(bn_bias_ptr + channel_id)
    
    # Compute element-wise operations
    for h in range(BLOCK_SIZE_H):
        for w in range(BLOCK_SIZE_W):
            h_idx = h_offset + h
            w_idx = w_offset + w
            
            if h_idx < height and w_idx < width:
                # Calculate memory index for current element
                mem_idx = batch_offset + channel_offset + h_idx * width + w_idx
                
                # Load inputs
                conv_out_val = tl.load(conv_out_ptr + mem_idx)
                residual_val = tl.load(residual_ptr + mem_idx)
                
                # Apply fused operations: scaling + addition + batch norm
                # (x * scale + residual - mean) / sqrt(var + eps) * weight + bias
                scaled = conv_out_val * scale_factor
                added = residual_val + scaled
                normalized = (added - mean) * rsqrt(var + 1e-05) * weight + bias
                
                # Store outputs
                tl.store(normalized_out_ptr + mem_idx, normalized)
                tl.store(added_out_ptr + mem_idx, added)

@torch.fx.wrap
def fused_scale_add_batch_norm(conv_out, scale_factor, residual, bn_mean, bn_var, bn_weight, bn_bias):
    # Get tensor shapes
    batch_size, n_channels, height, width = conv_out.shape
    
    # Determine block sizes for better GPU occupancy
    BLOCK_SIZE_N = 1  # Process one element at a time per thread for simplicity
    BLOCK_SIZE_H = 64
    BLOCK_SIZE_W = 64
    
    # Calculate grid dimensions
    num_batches = (batch_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_channels = (n_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_h_blocks = (height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    num_w_blocks = (width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Create output tensors
    normalized_out = torch.empty_like(conv_out)
    added_out = torch.empty_like(conv_out)
    
    # Launch kernel
    fused_scale_add_batch_norm_kernel[
        (num_batches, num_channels, num_h_blocks, num_w_blocks)
    ](
        conv_out_ptr=conv_out,
        scale_factor_ptr=scale_factor,
        residual_ptr=residual,
        bn_mean_ptr=bn_mean,
        bn_var_ptr=bn_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        normalized_out_ptr=normalized_out,
        added_out_ptr=added_out,
        n_channels=n_channels,
        height=height,
        width=width,
        batch_size=batch_size,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return normalized_out, added_out

# Replacement function
def replacement_func():
    return fused_scale_add_batch_norm