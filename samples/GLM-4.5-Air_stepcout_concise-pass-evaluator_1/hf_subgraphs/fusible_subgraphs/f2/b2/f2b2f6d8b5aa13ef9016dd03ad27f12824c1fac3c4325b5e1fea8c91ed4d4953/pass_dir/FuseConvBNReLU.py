import torch
import triton
import triton.language as tl

# Pattern matching function to match Conv2D + BatchNorm + LeakyReLU
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match Conv2D + BatchNorm + LeakyReLU pattern"""
    # Conv2D operation  
    tmp_6 = torch.conv2d(in_5, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    
    # BatchNorm operation
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # LeakyReLU operation
    tmp_8 = torch.nn.functional.leaky_relu(tmp_7, 0.01, True)
    
    return tmp_8

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

# Optimized fused Conv-BN-ReLU kernel using Triton
@triton.jit
def fused_conv_bn_relu_kernel(
    input_ptr,  # Input tensor (B, C_in, H, W)
    weight_ptr,  # Conv weights (C_out, C_in, KH, KW)
    running_mean_ptr,  # BN running mean (C_out,)
    running_var_ptr,  # BN running var (C_out,)
    weight_ptr_bn,  # BN weight (C_out,)
    bias_ptr_bn,  # BN bias (C_out,)
    output_ptr,  # Output tensor (B, C_out, H, W)
    B: tl.constexpr,  # Batch size
    C_in: tl.constexpr,  # Input channels
    C_out: tl.constexpr,  # Output channels  
    H: tl.constexpr,  # Input height
    W: tl.constexpr,  # Input width
    OH: tl.constexpr,  # Output height
    OW: tl.constexpr,  # Output width
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    eps: tl.constexpr,
    negative_slope: tl.constexpr,
):
    # Get program ID for 2D output grid
    pid = tl.program_id(0)
    batch_idx = pid // (OH * OW)
    out_h_idx = (pid // OW) % OH
    out_w_idx = pid % OW
    
    # Only process if within batch bounds
    if batch_idx >= B:
        return
    
    # Load BN parameters for all output channels
    c_out_idx = tl.arange(0, C_out)
    running_mean = tl.load(running_mean_ptr + c_out_idx, mask=c_out_idx < C_out)
    running_var = tl.load(running_var_ptr + c_out_idx, mask=c_out_idx < C_out)
    bn_weight = tl.load(weight_ptr_bn + c_out_idx, mask=c_out_idx < C_out)
    bn_bias = tl.load(bias_ptr_bn + c_out_idx, mask=c_out_idx < C_out)
    
    # Compute BN scaling parameters
    bn_scale = bn_weight / tl.sqrt(running_var + eps)
    bn_bias_adjusted = bn_bias - running_mean * bn_scale
    
    # Process input channels in chunks
    result = tl.zeros((C_out,), dtype=tl.float32)
    
    for c_in_idx in range(0, C_in):
        # Calculate input spatial indices with padding
        in_h_start = out_h_idx * stride_h - pad_h
        in_w_start = out_w_idx * stride_w - pad_w
        
        # Check if the convolution patch is within bounds
        h_valid = (in_h_start >= 0) and (in_h_start + KH <= H)
        w_valid = (in_w_start >= 0) and (in_w_start + KW <= W)
        if h_valid and w_valid:
            
            # Load convolution weights for this input channel
            weight_offset = (c_out_idx[:, None] * C_in * KH * KW + 
                           c_in_idx * KH * KW + 
                           tl.arange(0, KH)[None, :] * KW + 
                           tl.arange(0, KW)[None, :])
            conv_weights = tl.load(weight_ptr + weight_offset, 
                                 mask=c_out_idx[:, None] < C_out)
            
            # Load input patch
            input_offset = (batch_idx * C_in * H * W + 
                          c_in_idx * H * W + 
                          tl.arange(0, KH) * W + 
                          tl.arange(0, KW))
            input_patch = tl.load(input_ptr + input_offset)
            
            # Compute convolution for this input channel
            conv_result = (conv_weights * input_patch).sum()
            result += conv_result
    
    # Apply batch normalization
    bn_result = result * bn_scale + bn_bias_adjusted
    
    # Apply leaky ReLU
    relu_result = tl.where(bn_result > 0, bn_result, bn_result * negative_slope)
    
    # Store result
    output_offset = batch_idx * C_out * OH * OW + c_out_idx * OH * OW + out_h_idx * OW + out_w_idx
    tl.store(output_ptr + output_offset, relu_result, mask=c_out_idx < C_out)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv_bn_relu(in_0, in_1, in_2, in_3, in_4, in_5):
    """Fused Conv2D-BatchNorm-LeakyReLU operation"""
    # Get tensor shapes
    B, C_in, H, W = in_5.shape
    C_out, _, KH, KW = in_4.shape
    
    # For stride=1, pad=1, dilation=1: output dimensions = input dimensions
    OH = H
    OW = W
    
    # Calculate total number of output elements
    total_elements = B * OH * OW
    
    # Launch Triton kernel - one program per output position and batch element
    grid_size = (total_elements,)
    
    # Create output tensor
    output = torch.empty((B, C_out, OH, OW), dtype=in_5.dtype, device=in_5.device)
    
    # Launch Triton kernel
    fused_conv_bn_relu_kernel[grid_size](
        input_ptr=in_5,
        weight_ptr=in_4,
        running_mean_ptr=in_0,
        running_var_ptr=in_1,
        weight_ptr_bn=in_3,
        bias_ptr_bn=in_2,
        output_ptr=output,
        B=B,
        C_in=C_in,
        C_out=C_out,
        H=H,
        W=W,
        OH=OH,
        OW=OW,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        eps=1e-05,
        negative_slope=0.01,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_conv_bn_relu