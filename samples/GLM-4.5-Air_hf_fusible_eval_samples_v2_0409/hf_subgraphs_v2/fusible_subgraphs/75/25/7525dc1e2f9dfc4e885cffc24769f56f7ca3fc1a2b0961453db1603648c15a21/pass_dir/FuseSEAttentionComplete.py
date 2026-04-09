import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    """Match the complete SE attention computation: Conv2D + HardSigmoid + Element-wise multiply + Adaptive avg pool2d + Flatten"""
    # Match the exact operations from model.py
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardsigmoid(tmp_2, False)
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    # Dropout with p=0.0 is effectively a no-op, so we don't include it in the pattern
    return tmp_6

# Argument extraction function  
def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the fused kernel"""
    return (in_0, in_1, in_2, in_3)

# Final optimized Triton kernel with mathematical optimization
@triton.jit
def final_optimized_se_attention_kernel(
    out_ptr,            # [B, 1024] - output tensor
    bias_ptr,           # [1024] - conv bias 
    weight_ptr,         # [1024, 1024, 1, 1] - conv weights  
    in_2_ptr,           # [B, 1024, H, W] - input feature map
    in_3_ptr,           # [B, 1024, 1, 1] - conv input
    batch_size,         # batch dimension  
    channels,           # feature channels (1024)
    in_2_height,        # spatial height of in_2
    in_2_width,         # spatial width of in_2,
):
    """
    Final optimized kernel that applies mathematical transformation:
    Original: conv → hardsigmoid → (in_2 * hardsigmoid_val).mean()
    Optimized:  conv → hardsigmoid → hardsigmoid_val * mean(in_2)
    This eliminates expensive element-wise multiplication with broadcasting!
    """
    # Each program handles one batch and one channel combination
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Check bounds to prevent out-of-access
    if batch_idx >= batch_size or channel_idx >= channels:
        return
    
    # Calculate base pointers
    batch_offset = batch_idx * channels
    channel_offset = channel_idx
    flattened_idx = batch_offset + channel_offset
    
    # 1. Load conv parameters (cached in registers)
    bias_val = tl.load(bias_ptr + channel_offset)
    conv_weight = tl.load(weight_ptr + (channel_idx * channels + channel_offset))
    
    # 2. Load in_3 value for conv input
    in_3_val = tl.load(in_3_ptr + batch_offset + channel_offset)
    
    # 3. Compute conv2d output and apply HardSigmoid
    conv_output = in_3_val * conv_weight + bias_val
    # Optimized HardSigmoid: max(0, min(1, x * 3 + 3)) * 1/6
    hard_sigmoid_val = tl.maximum(0.0, tl.minimum(1.0, conv_output * 3.0 + 3.0)) / 6.0
    
    # 4. Compute mean of in_2 for this batch-channel (key optimization!)
    # This eliminates the expensive element-wise multiplication with broadcasting
    in_2_base_offset = (batch_idx * channels + channel_idx) * in_2_height * in_2_width
    spatial_sum = 0.0
    total_elements = in_2_height * in_2_width
    
    # Simple, safe loop over spatial dimensions (no vectorized loads)
    for h in range(in_2_height):
        for w in range(in_2_width):
            spatial_sum += tl.load(in_2_ptr + in_2_base_offset + h * in_2_width + w)
    
    mean_in_2 = spatial_sum / total_elements
    
    # 5. Final computation: hard_sigmoid * mean(in_2)
    # This mathematically equivalent to (in_2 * hard_sigmoid_val).mean()
    # but eliminates the expensive broadcasting multiplication!
    final_val = mean_in_2 * hard_sigmoid_val
    
    # Store result
    tl.store(out_ptr + flattened_idx, final_val)

@torch.fx.wrap  
def fused_se_attention_triton(in_0, in_1, in_2, in_3):
    """Wrapper function to launch the final optimized fused kernel"""
    batch_size = in_2.shape[0]
    channels = in_2.shape[1]  # Always 1024
    in_2_height = in_2.shape[2]
    in_2_width = in_2.shape[3]
    
    # Allocate output tensor
    output = torch.empty((batch_size, channels), dtype=in_2.dtype, device=in_2.device)
    
    # Use grid of (batch_size, channels) - optimal for parallelism
    grid = (batch_size, channels)
    
    # Launch final optimized kernel with mathematical transformation
    final_optimized_se_attention_kernel[grid](
        out_ptr=output,
        bias_ptr=in_0,
        weight_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        batch_size=batch_size,
        channels=channels,
        in_2_height=in_2_height,
        in_2_width=in_2_width,
    )
    
    return output

# Replacement function (returns function reference, not a call)
def replacement_func():
    return fused_se_attention_triton