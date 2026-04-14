import torch
import triton
import triton.language as tl
import math

# Pattern matching function - match the full computation that produces tmp_4
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.sigmoid(conv2d)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    tmp_4 = in_2 * tmp_3
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)  # Return all inputs as they are needed for the computation

# Optimized fused kernel for entire computation: conv2d + sigmoid + interpolate + multiply
@triton.jit
def fused_full_computation_kernel(
    x_ptr,                           # Input tensor pointer [1, 960, 1, 4]
    weight_ptr,                      # Weight tensor pointer [128, 960, 1, 1]
    mul_ptr,                         # Multiplication tensor pointer [1, 128, 64, 128]
    out_ptr,                         # Output tensor pointer [1, 128, 64, 128]
    batch,                           # Batch size = 1
    in_channels,                     # Input channels = 960
    out_channels,                    # Output channels = 128
    in_height,                       # Input height = 1
    in_width,                        # Input width = 4
    out_height,                      # Final output height = 64
    out_width,                       # Final output width = 128
    weight_height,                   # Weight height = 1
    weight_width,                    # Weight width = 1
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program processes one final output position (H, W)
    if pid >= out_height * out_width:
        return
    
    # Calculate final output position
    out_pos = pid
    final_h = out_pos // out_width
    final_w = out_pos % out_width
    
    # Process all channels for this final position
    for c in range(out_channels):
        # Step 1: Conv2D + Sigmoid for this channel and original position
        # Original input position is at height 0, width=(final_w * (in_width-1)) / (out_width-1)
        orig_w = (final_w * (in_width - 1)) / (out_width - 1) if out_width > 1 else 0.0
        
        # Convolution computation for output channel c
        acc = 0.0
        for ic in range(in_channels):
            # Weight for this input channel and output channel
            weight_idx = c * in_channels * weight_height * weight_width + ic * weight_height * weight_width
            weight = tl.load(weight_ptr + weight_idx)
            
            # Input position: height 0, width orig_w (integer position for load)
            orig_w0 = int(orig_w)
            orig_w1 = min(orig_w0 + 1, in_width - 1) if in_width > 1 else orig_w0
            
            if orig_w0 < in_width:
                in_idx = ic * (batch * in_height * in_width) + (0 * in_width + orig_w0)
                x_val = tl.load(x_ptr + in_idx)
                acc += x_val * weight
        
        if orig_w1 < orig_w0 + 1 and orig_w1 < in_width:
            w_weight_in = orig_w - orig_w0
            w_weight_out = 1.0 - w_weight_in
            in_idx = ic * (batch * in_height * in_width) + (0 * in_width + orig_w1)
            x_val = tl.load(x_ptr + in_idx)
            acc += w_weight_in * x_val * weight
        
        # Step 2: Apply sigmoid activation
        sigmoid_val = 1.0 / (1.0 + tl.exp(-acc))
        
        # Step 3: Multiply with corresponding multiplication value
        out_idx = c * (batch * out_height * out_width) + (final_h * out_width + final_w)
        mul_val = tl.load(mul_ptr + out_idx)
        result = mul_val * sigmoid_val
        
        # Store final result
        tl.store(out_ptr + out_idx, result)

@torch.fx.wrap
def fused_full_computation(in_0, in_1, in_2):
    # Get tensor shapes
    x_shape = in_1.shape  # [1, 960, 1, 4]
    weight_shape = in_0.shape  # [128, 960, 1, 1]
    mul_shape = in_2.shape  # [1, 128, 64, 128]
    
    # Output shape: [1, 128, 64, 128]
    out_shape = mul_shape
    out = torch.empty(out_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Kernel launch parameters
    batch = x_shape[0]
    in_channels = x_shape[1]
    out_channels = weight_shape[0]
    in_height = x_shape[2]
    in_width = x_shape[3]
    final_out_height = mul_shape[2]
    final_out_width = mul_shape[3]
    weight_height = weight_shape[2]
    weight_width = weight_shape[3]
    
    BLOCK_SIZE = 256  # Number of output pixels to process per program
    
    # Launch kernel
    grid = (final_out_height * final_out_width,)  # One program per final output position
    
    fused_full_computation_kernel[grid](
        in_1,
        in_0,
        in_2,
        out,
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
        final_out_height,
        final_out_width,
        weight_height,
        weight_width,
        BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_full_computation