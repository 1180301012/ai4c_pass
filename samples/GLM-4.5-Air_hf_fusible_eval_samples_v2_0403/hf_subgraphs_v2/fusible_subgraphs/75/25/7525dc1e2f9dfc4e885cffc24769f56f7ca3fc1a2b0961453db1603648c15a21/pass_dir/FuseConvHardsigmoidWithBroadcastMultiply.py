import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Conv2D with 1x1 kernel producing [N, C, 1, 1] output
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    # Harsigmoid activation keeps the same shape [N, C, 1, 1]
    tmp_3 = torch.nn.functional.hardsigmoid(conv2d, False)
    
    # Element-wise multiplication with broadcasting: [N, C, H, W] * [N, C, 1, 1]
    tmp_4 = in_2 * tmp_3
    
    # Adaptive average pooling reduces to [N, C, 1, 1]
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    
    # Flatten to [N, C]
    tmp_6 = tmp_5.flatten(1, -1)
    
    # Dropout with rate 0.0 (no-op)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    
    return tmp_7

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_optimization_kernel(
    bias_ptr,
    weight_ptr,
    conv_input_ptr,
    spatial_input_ptr,
    out_ptr,
    N, C_in, C_out, H, W,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Complete fusion of: conv2d + hardsigmoid + multiply + adaptive_avg_pool2d + flatten
    """
    pid = tl.program_id(0)
    
    # Each program handles one output channel
    batch_idx = pid // C_out
    channel_idx = pid % C_out
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + channel_idx)
    
    # Load weights for this output channel: [C_in] weights
    weight_offset = batch_idx * C_out * C_in + channel_idx * C_in
    weights = tl.load(weight_ptr + weight_offset + tl.arange(0, C_in))
    
    # Load conv input for this batch: [C_in] input channels
    conv_input_offset = batch_idx * C_in
    conv_inputs = tl.load(conv_input_ptr + conv_input_offset + tl.arange(0, C_in))
    
    # Compute conv: sum(weights * inputs) + bias
    conv_val = tl.sum(weights * conv_inputs) + bias
    
    # Harsigmoid: max(0, min(1, conv_val * 0.2 + 0.5))
    hardsigmoid = tl.maximum(0.0, tl.minimum(1.0, conv_val * 0.2 + 0.5))
    
    # Compute mean of spatial input for this batch and channel
    # Load all spatial data for this batch and channel: [H, W]
    spatial_offset = batch_idx * C_in * H * W + channel_idx * H * W
    
    # Efficient reduction to compute mean
    spatial_sum = 0.0
    spatial_count = H * W
    
    # Vectorized loading for better performance
    vec_size = min(BLOCK_SIZE_N, W)
    remaining = W
    
    for h in range(H):
        base_offset = spatial_offset + h * W
        
        # Process in chunks for vectorization
        w = 0
        while w < remaining:
            chunk_size = min(vec_size, remaining - w)
            offsets = base_offset + w + tl.arange(0, chunk_size)
            spatial_data = tl.load(spatial_input_ptr + offsets)
            spatial_sum += tl.sum(spatial_data)
            w += chunk_size
    
    mean_spatial_input = spatial_sum / spatial_count
    
    # Final computation: harsigmoid * mean_spatial_input
    result = hardsigmoid * mean_spatial_input
    
    # Store result
    tl.store(out_ptr + pid, result)

@torch.fx.wrap
def fused_conv_hardsigmoid_multiply_pool(in_0, in_1, in_2, in_3):
    """
    Optimized fusion of conv2d + hardsigmoid + multiply + pooling
    
    Mathematical insight:
    adaptive_avg_pool2d(in_2 * hardsigmoid(conv2d(in_3, in_1, in_0)), 1)
    = hardsigmoid(conv2d(in_3, in_1, in_0)) * adaptive_avg_pool2d(in_2, 1)
    
    This allows us to:
    1. Compute mean of in_2 efficiently (single pooling operation)
    2. Compute conv2d + harsigmoid efficiently
    3. Multiply the results (simple element-wise operation)
    """
    # Step 1: Compute mean of in_2 using efficient torch implementation
    mean_in_2 = torch.nn.functional.adaptive_avg_pool2d(in_2, 1).flatten(1, -1)
    
    # Step 2: Compute conv2d + harsigmoid using optimized triton kernel
    N, C_in, H_in, W_in = in_3.shape
    C_out = in_0.shape[0]
    
    # Reshape inputs for efficient processing
    in_0_flat = in_0  # [C_out]
    in_1_flat = in_1.view(C_out, C_in)  # [C_out, C_in] 
    in_3_flat = in_3.view(N, C_in)  # [N, C_in]
    
    # Output will be [N * C_out]
    out_flat = torch.empty(N * C_out, dtype=in_2.dtype, device=in_2.device)
    
    # Launch triton kernel
    grid = (N * C_out,)
    BLOCK_SIZE = min(1024, C_in)  # Optimized block size
    
    conv_hardsigmoid_kernel[grid](
        in_0_flat,
        in_1_flat,
        in_3_flat,
        out_flat,
        N, C_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to [N, C_out] for multiplication
    conv_hardsigmoid_out = out_flat.view(N, C_out)
    
    # Step 3: Multiply results (element-wise operation)
    result = conv_hardsigmoid_out * mean_in_2
    
    return result

def replacement_func():
    return fused_conv_hardsigmoid_multiply_pool