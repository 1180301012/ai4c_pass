import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4):
    # Conv2D + LayerNorm + ReLU fusion pattern
    tmp_4 = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_5 = torch.nn.functional.layer_norm(tmp_4, (in_0.shape[0], 1, 1), in_3, in_2, 1e-05)
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    return (tmp_6,)

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)

@triton.jit
def fused_conv_norm_relu_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    batch_size_hw,
    in_channels,
    out_channels,
    spatial_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Triton kernel for fused Conv2D -> LayerNorm -> ReLU
    # We treat this as batched matrix multiplication where each element is treated as a batch
    
    pid = tl.program_id(0)
    
    # Tile sizes
    m_range = tl.arange(0, BLOCK_SIZE_M)
    n_range = tl.arange(0, BLOCK_SIZE_N)
    k_range = tl.arange(0, BLOCK_SIZE_K)
    
    # Program's range in batch dimension (each program processes multiple batch*channel positions)
    batch_start = pid * BLOCK_SIZE_M
    batch_end = min(batch_start + BLOCK_SIZE_M, batch_size_hw)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    eps = 1e-05
    
    # Matrix multiplication: output[bhw, cout] = sum_cin (input[bhw, cin] * weight[cout, cin]) + bias[cout]
    for k in range(0, in_channels, BLOCK_SIZE_K):
        # Compute bounds for current k tile
        k_block_start = k
        k_block_end = min(k + BLOCK_SIZE_K, in_channels)
        k_tile_size = k_block_end - k_block_start
        
        # Load input tile: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        input_offsets = (batch_start + m_range.reshape(-1, 1)) * in_channels + (k_block_start + k_range.reshape(1, -1))
        input_tile = tl.load(input_ptr + input_offsets.to(tl.int64), 
                            mask=(batch_start + m_range.reshape(-1, 1)) < batch_end,
                            other=0.0)
        
        # Load weight tile: [BLOCK_SIZE_N, BLOCK_SIZE_K]
        weight_offsets = n_range.reshape(-1, 1) * in_channels + (k_block_start + k_range.reshape(1, -1))
        weight_tile = tl.load(weight_ptr + weight_offsets.to(tl.int64), 
                            mask=n_range < out_channels,
                            other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(input_tile, weight_tile, trans_b=True)
    
    # Add bias
    bias_offsets = (batch_start + m_range.reshape(-1, 1)) * out_channels + n_range.reshape(1, -1)
    bias_tile = tl.load(bias_ptr + bias_offsets.to(tl.int64) + n_range.reshape(1, -1),
                       mask=(batch_start + m_range.reshape(-1, 1)) < batch_end and n_range.reshape(1, -1) < out_channels,
                       other=0.0)
    accumulator += bias_tile
    
    # Now compute statistics for LayerNorm across all spatial positions for each channel
    # Since we have [C, 1, 1] LayerNorm, we need to compute mean and variance per channel
    # across all batch and spatial positions
    
    # Initialize sum of squares for variance computation
    sum_vals = tl.zeros(BLOCK_SIZE_N, dtype=tl.float32)
    sum_squares = tl.zeros(BLOCK_SIZE_N, dtype=tl.float32)
    
    # Sum across batch dimension for each output channel
    for b_idx in range(BLOCK_SIZE_M):
        if batch_start + b_idx < batch_end:
            row_vals = accumulator[b_idx, :]
            sum_vals += row_vals
            sum_squares += row_vals * row_vals
    
    # Normalize by total number of elements (batch_size * spatial_size)
    total_elements = spatial_size
    mean = sum_vals / total_elements
    var = sum_squares / total_elements - mean * mean
    
    # Load LayerNorm parameters and apply normalization
    norm_weight = tl.load(norm_weight_ptr + n_range.reshape(-1),
                         mask=n_range < out_channels,
                         other=0.0)
    norm_bias = tl.load(norm_bias_ptr + n_range.reshape(-1),
                        mask=n_range < out_channels,
                        other=0.0)
    
    # Apply LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized = (accumulator - mean.reshape(1, -1)) * inv_std.reshape(1, -1) * norm_weight.reshape(-1, 1) + norm_bias.reshape(-1, 1)
    
    # Apply ReLU activation
    relu_result = tl.maximum(normalized, 0.0)
    
    # Store final result
    output_offsets = (batch_start + m_range.reshape(-1, 1)) * out_channels + n_range.reshape(1, -1)
    output_mask = (batch_start + m_range.reshape(-1, 1)) < batch_end
    
    tl.store(output_ptr + output_offsets.to(tl.int64), relu_result, mask=output_mask)

@torch.fx.wrap
def fused_conv_norm_relu(in_0, in_1, in_2, in_3, in_4):
    """
    Fused Conv2D + LayerNorm + ReLU implementation
    in_0: conv bias
    in_1: conv weight
    in_2: layer norm weight
    in_3: layer norm bias
    in_4: input tensor
    """
    batch_size, in_channels, height, width = in_4.shape
    out_channels = in_0.shape[0]
    
    # Create output tensor
    out = torch.empty((batch_size, out_channels, height, width), dtype=in_4.dtype, device=in_4.device)
    
    # For 1x1 conv + LayerNorm([C, 1, 1]) fusion, we can optimize significantly
    # We reshape input to [batch_size * height * width, in_channels] for matrix multiplication
    input_flat = in_4.reshape(-1, in_channels)
    output_flat = out.reshape(-1, out_channels)
    
    # Choose optimal block sizes based on tensor dimensions
    total_elements = batch_size * height * width
    
    # Block size for batch dimension (program_id dimension)
    if total_elements < 1024:
        BLOCK_SIZE_M = 64
    elif total_elements < 4096:
        BLOCK_SIZE_M = 128
    else:
        BLOCK_SIZE_M = 256
    
    # Block size for output channels (inner dimension)
    if out_channels <= 128:
        BLOCK_SIZE_N = out_channels
    elif out_channels <= 512:
        BLOCK_SIZE_N = 128
    else:
        BLOCK_SIZE_N = 256
    
    # Block size for matrix multiplication (shared dimension)
    BLOCK_SIZE_K = 32 if in_channels >= 32 else in_channels
    
    # Calculate grid size
    grid_size = (total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch the fused kernel
    fused_conv_norm_relu_kernel[grid_size](
        output_ptr=output_flat,
        input_ptr=input_flat,
        weight_ptr=in_1.reshape(out_channels, in_channels),  # Reshape weight from [C_out, C_in, 1, 1] to [C_out, C_in]
        bias_ptr=in_0,
        norm_weight_ptr=in_3,  # LayerNorm weight
        norm_bias_ptr=in_2,    # LayerNorm bias
        batch_size_hw=total_elements,
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_size=height * width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

def replacement_func():
    return fused_conv_norm_relu