import torch
import triton
import triton.language as tl
import math

def pattern(conv_input, conv_weight, conv_bias, input1, input2):
    # Convolution
    tmp_6 = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)  # groups=1 for standard conv2d
    conv_weight = conv_bias = None
    # First addition
    tmp_7 = input2 + tmp_6
    tmp_6 = None
    # Second addition  
    tmp_8 = tmp_7 + input1
    tmp_7 = None
    return tmp_8

def replacement_args(conv_input, conv_weight, conv_bias, input1, input2):
    return (conv_input, conv_weight, conv_bias, input1, input2)

@triton.jit
def fused_conv_add_add_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    input1_ptr,
    input2_ptr,
    out_ptr,
    N,  # batch size
    C,  # output channels
    H,  # height
    W,  # width
    IC, # input channels
    KH: tl.constexpr,  # kernel height (1)
    KW: tl.constexpr,  # kernel width (1)
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2) if tl.num_programs(2) > 1 else 0
    pid_w = tl.program_id(3) if tl.num_programs(3) > 1 else 0
    
    # Compute coordinates
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W
    h_end = min(h_start + BLOCK_SIZE_H, H)
    w_end = min(w_start + BLOCK_SIZE_W, W)
    
    # Initialize output accumulator
    acc = 0.0
    
    # For 1x1 convolution, we can simplify significantly
    # Load bias directly for the output position
    bias_val = tl.load(bias_ptr + pid_c)
    
    # Process the 1x1 convolution
    x_offset = pid_n * IC * H * W + pid_c * H * W + h_start * W + w_start
    weight_offset = pid_c * IC * KH * KW
    
    conv_val = bias_val
    
    # For 1x1 conv, we need to multiply input channels
    for ic in range(IC):
        # Get input value at this location
        input_val = tl.load(x_ptr + x_offset + ic * H * W, other=0.0)
        # Get weight value (1x1 conv so IC weights per output channel)
        weight_val = tl.load(weight_ptr + weight_offset + ic * KH * KW + pid_c * IC * KH * KW, other=0.0)
        conv_val += input_val * weight_val
    
    # Load the two input tensors for addition
    input1_val = tl.load(input1_ptr + pid_n * C * H * W + pid_c * H * W + h_start * W + w_start, other=0.0)
    input2_val = tl.load(input2_ptr + pid_n * C * H * W + pid_c * H * W + h_start * W + w_start, other=0.0)
    
    # Fused computation: conv + input2 + input1
    result = conv_val + input2_val + input1_val
    
    # Store result
    out_offset = pid_n * C * H * W + pid_c * H * W + h_start * W + w_start
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            offset = out_offset + (h - h_start) * W + (w - w_start)
            tl.store(out_ptr + offset, result)

@torch.fx.wrap
def fused_conv_add_add(x, weight, bias, input1, input2):
    N, IC, H, W = x.shape
    OC = weight.shape[0]  # Output channels from weight shape
    
    # Determine output shape (should be same as input for 1x1 conv)
    out = torch.empty((N, OC, H, W), device=x.device, dtype=x.dtype)
    
    # Calculate grid size
    # Use 2D grid for spatial dimensions, but may need to adjust for performance
    grid = (N, OC, 1, 1)  # Simple grid for now
    
    # Launch kernel
    fused_conv_add_add_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        input1_ptr=input1,
        input2_ptr=input2,
        out_ptr=out,
        N=N,
        C=OC,
        H=H,
        W=W,
        IC=IC,
        KH=1,  # kernel height
        KW=1,  # kernel width
        BLOCK_SIZE_N=1,
        BLOCK_SIZE_C=1,
        BLOCK_SIZE_H=H,
        BLOCK_SIZE_W=W,
    )
    
    return out

def replacement_func():
    return fused_conv_add_add