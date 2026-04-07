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
    BLOCK_SIZE: tl.constexpr,
):
    """
    Complete fusion of: conv2d + harsigmoid + multiply + adaptive_avg_pool2d + flatten
    """
    pid = tl.program_id(0)
    
    # Each program handles one output channel
    batch_idx = pid // C_out
    channel_idx = pid % C_out
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + channel_idx)
    
    # Compute convolution with loop over channels (since C_in is not constant)
    conv_val = bias  # Start with bias
    for c in range(C_in):
        # Load weight for this channel
        weight_offset = batch_idx * C_out * C_in + channel_idx * C_in + c
        weight = tl.load(weight_ptr + weight_offset)
        
        # Load input for this channel
        conv_input_offset = batch_idx * C_in + c
        conv_input = tl.load(conv_input_ptr + conv_input_offset)
        
        # Add to convolution sum
        conv_val += weight * conv_input
    
    # Harsigmoid: max(0, min(1, conv_val * 0.2 + 0.5))
    hardsigmoid = tl.maximum(0.0, tl.minimum(1.0, conv_val * 0.2 + 0.5))
    
    # Compute mean of spatial input for this batch and channel
    spatial_offset = batch_idx * C_in * H * W + channel_idx * H * W
    spatial_sum = 0.0
    spatial_count = H * W
    
    # Simple loop over spatial dimensions for safety
    for h in range(H):
        for w in range(W):
            spatial_idx = spatial_offset + h * W + w
            spatial_data = tl.load(spatial_input_ptr + spatial_idx)
            spatial_sum += spatial_data
    
    mean_spatial_input = spatial_sum / spatial_count
    
    # Final computation: harsigmoid * mean_spatial_input
    result = hardsigmoid * mean_spatial_input
    
    # Store result
    tl.store(out_ptr + pid, result)

@torch.fx.wrap
def fused_conv_hardsigmoid_multiply_pool(in_0, in_1, in_2, in_3):
    """
    Optimized implementation using mathematical transformation to avoid
    illegal memory access and improve performance.
    
    Key insight: adaptive_avg_pool2d(in_2 * hardsigmoid(conv2d(...)), 1)
    = hardsigmoid(conv2d(...)) * adaptive_avg_pool2d(in_2, 1)
    """
    # Step 1: Compute conv2d + harsigmoid part
    conv_output = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    hardsigmoid_output = torch.nn.functional.hardsigmoid(conv_output, False)
    
    # Step 2: Compute mean of spatial input separately 
    mean_in_2 = torch.nn.functional.adaptive_avg_pool2d(in_2, 1).flatten(1, -1)
    
    # Step 3: Multiply results (broadcasting handles the rest)
    result = hardsigmoid_output.flatten(1, -1) * mean_in_2
    
    return result

def replacement_func():
    return fused_conv_hardsigmoid_multiply_pool