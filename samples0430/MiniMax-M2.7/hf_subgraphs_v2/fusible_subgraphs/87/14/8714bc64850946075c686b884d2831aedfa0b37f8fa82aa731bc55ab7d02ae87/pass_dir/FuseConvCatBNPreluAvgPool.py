"""
Fused kernel for Conv2d + Cat + BatchNorm + PReLU + AdaptiveAvgPool2d + View
This pass fuses the entire computation pipeline into a single GPU kernel.
"""
import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Match the pattern: conv2d -> cat -> batch_norm -> prelu -> adaptive_avg_pool2d -> view
    """
    # Conv2d with depthwise separable convolution
    # torch.conv2d(input, weight, bias=None, stride=(1,1), padding=(4,4), dilation=(4,4), groups=64)
    # Note: groups=64 means depthwise: each input channel gets its own filter
    conv2d = torch.conv2d(in_7, in_5, None, (1, 1), (4, 4), (4, 4), 64)
    
    # Cat along dim=1 (channel dimension)
    tmp_7 = torch.cat([in_6, conv2d], 1)
    
    # BatchNorm: torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 0.001)
    
    # PReLU: torch.prelu(input, weight) - channel-wise activation
    tmp_9 = torch.prelu(tmp_8, in_0)
    
    # AdaptiveAvgPool2d to 1x1
    tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_9, 1)
    
    # View reshape (batch, channels)
    tmp_11 = tmp_10.view(in_6.shape[0], 128)
    
    return tmp_9, tmp_11


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """Extract arguments needed for the replacement kernel"""
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_conv_bn_prelu_avgpool_kernel(
    # Conv weights: [64, 1, 3, 3] - depthwise conv
    weight_ptr,
    # BN parameters
    bn_mean_ptr, bn_var_ptr, bn_gamma_ptr, bn_beta_ptr,
    # PReLU weight: [128]
    prelu_weight_ptr,
    # Input tensors (original and for cat)
    x_ptr, x_cat_ptr,
    # Output tensor
    output_ptr, pooled_output_ptr,
    # Strides
    x_stride, x_cat_stride,
    # Output strides
    out_stride, pool_stride,
    # Dimensions
    N,  # batch * height * width
    C,  # 64 input channels, output will be 128
    H, W,  # spatial dimensions
    pool_out_size,  # output channel size (128)
    # Scalar parameters
    eps: tl.constexpr,
    bn_momentum: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Depthwise conv2d with dilation
    2. Cat along channel dim
    3. BatchNorm
    4. PReLU
    5. AdaptiveAvgPool2d (reduce spatial to 1x1)
    """
    # Program ID for output channels
    pid_c = tl.program_id(0)
    # Program ID for batch dimension
    pid_b = tl.program_id(1)
    
    # Calculate output channel offset
    if pid_c >= C:  # This is the concatenated part (channels C to 2*C)
        channel_type = 1
        c_out = pid_c - C
        c_in = c_out  # Same channel from input for depthwise
    else:  # This is the original tensor passthrough (channels 0 to C)
        channel_type = 0
        c_out = pid_c
        c_in = c_out
    
    # Calculate output offset
    out_offset = pid_b * pool_out_size + pid_c
    out_ptrs = output_ptr + out_offset
    
    # For adaptive avg pool to 1x1, we accumulate all spatial positions
    accum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Iterate over all spatial positions
    for h in range(H):
        for w in range(W):
            # Calculate input offsets
            input_offset = pid_b * x_stride + c_in * H * W + h * W + w
            input_cat_offset = pid_b * x_cat_stride + c_in * H * W + h * W + w
            
            if channel_type == 0:
                # Original tensor passthrough (channel-wise)
                x_val = tl.load(x_ptr + input_cat_offset)
                conv_val = x_val
            else:
                # Depthwise conv with dilation on the x tensor
                # For dilated conv, we read with dilation factor 4
                # Dilation means we sample every 4th pixel
                # Padding of 4 means input effectively larger
                # Since dilation=4 and we have stride=1, padding accounts for this
                
                # For dilated conv on x_ptr (x_83):
                # Input spatial size = H x W
                # With padding=4, dilation=4, kernel_size=3
                # The output at (h, w) uses input[(h*1):(h*1+3), (w*1):(w*1+3)] * dilated_kernel
                # With dilation=4, we sample at offsets 0, 4, 8
                # With H=48, W=48, padding=4 means input region is 48+8=56
                # But the input is already H=48, W=48, so we need to handle boundary
                
                conv_val = tl.float32(0.0)
                
                # Depthwise separable conv: each output channel uses its own 3x3 kernel
                # kernel_offset = c_in * 9 (since each weight is 1x3x3)
                for kh in range(3):
                    for kw in range(3):
                        # Calculate dilated input position
                        # With dilation=4, the kernel positions are at 0, 4, 8 offsets
                        h_in = h + (kh - 1) * 4  # centered, with dilation
                        w_in = w + (kw - 1) * 4
                        
                        # Check bounds (handle padding implicitly with clamp)
                        if 0 <= h_in < H and 0 <= w_in < W:
                            input_pos = pid_b * x_stride + c_in * H * W + h_in * W + w_in
                            weight_pos = c_in * 9 + kh * 3 + kw
                            weight_val = tl.load(weight_ptr + weight_pos)
                            inp_val = tl.load(x_ptr + input_pos)
                            conv_val = conv_val + inp_val * weight_val
                
                # Load original tensor for cat
                x_val = tl.load(x_ptr + input_offset)
            
            # Cat: select based on channel type
            cat_val = x_val if channel_type == 0 else conv_val
            
            # Load BN parameters for this channel
            mean = tl.load(bn_mean_ptr + c_out)
            var = tl.load(bn_var_ptr + c_out)
            gamma = tl.load(bn_gamma_ptr + c_out)
            beta = tl.load(bn_beta_ptr + c_out)
            
            # BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
            inv_std = tl.rsqrt(var + eps)
            bn_val = (cat_val - mean) * inv_std * gamma + beta
            
            # PReLU: max(0, x) + weight * min(0, x)
            weight = tl.load(prelu_weight_ptr + c_out)
            prelu_val = tl.where(bn_val > 0, bn_val, bn_val * weight)
            
            # Accumulate for avg pool
            accum = accum + prelu_val
    
    # Average pooling: divide by total spatial elements
    pool_val = accum / (H * W)
    
    # Store output and pooled output
    tl.store(output_ptr + out_offset, prelu_val, mask=None)
    tl.store(pooled_output_ptr + pid_b * 128 + pid_c, pool_val)


@torch.fx.wrap
def fused_conv_bn_prelu_avgpool(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Wrapper function that launches the fused kernel.
    Handles the fused computation: Conv2d + Cat + BN + PReLU + AvgPool + View
    """
    # Get input dimensions
    batch, channels, H, W = in_7.shape  # in_7 is [B, 64, H, W]
    assert channels == 64, f"Expected 64 input channels, got {channels}"
    
    # BN parameters (running mean, var, weight, bias)
    bn_mean = in_1  # [128]
    bn_var = in_2   # [128]
    bn_gamma = in_4  # [128] - BN weight (scale)
    bn_beta = in_3   # [128] - BN bias
    
    # PReLU weight
    prelu_weight = in_0  # [128]
    
    # Conv weight: [64, 1, 3, 3]
    conv_weight = in_5
    
    # Output channels after cat: 64 (original) + 64 (conv) = 128
    out_channels = 128
    
    # Output tensor: [batch, 128, H, W]
    output = torch.empty((batch, out_channels, H, W), device=in_7.device, dtype=in_7.dtype)
    
    # Pooled output tensor: [batch, 128]
    pooled_output = torch.empty((batch, out_channels), device=in_7.device, dtype=in_7.dtype)
    
    # Calculate strides
    x_stride = in_7.stride(0)  # batch stride
    
    # Grid configuration
    # Each program handles one output channel
    grid = (out_channels, batch, 1)
    
    # Launch fused kernel
    fused_conv_bn_prelu_avgpool_kernel[grid](
        conv_weight, 
        bn_mean, bn_var, bn_gamma, bn_beta,
        prelu_weight,
        in_7, in_6,  # x_ptr, x_cat_ptr (in_6 is original, in_7 is conv input)
        output, pooled_output,
        x_stride, in_6.stride(0),
        output.stride(0), pooled_output.stride(0),
        batch * H * W,  # N
        channels,  # C = 64
        H, W,
        out_channels,
        eps=0.001,
        bn_momentum=0.1,
    )
    
    return output, pooled_output


def replacement_func():
    return fused_conv_bn_prelu_avgpool