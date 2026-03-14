import torch
import triton
import triton.language as tl

# Pattern matching function - matches the conv + view + batch_norm + relu sequence
def pattern(in_0, in_1, in_2, in_3, in_4, in_9):
    """
    Match: conv2d + view + batch_norm + relu
    This mirrors the exact operations from model.py
    """
    # Conv2d with groups (depthwise convolution)
    tmp_4 = torch.conv2d(input=in_9, weight=in_4, groups=512)
    # View/reshape
    tmp_5 = tmp_4.view(1, 512, 64, 64)
    # Batch normalization
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    # ReLU activation
    tmp_7 = torch.nn.functional.relu(tmp_6, inplace=False)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_9):
    """Extract arguments needed for the fused kernel"""
    return (in_0, in_1, in_2, in_3, in_4, in_9)


# Optimized Triton kernel: Fused Depthwise Conv + BatchNorm + ReLU
@triton.jit
def fused_conv_bn_relu_kernel(
    # Input tensor (in_9)
    input_ptr, weight_ptr,
    # Batch norm parameters (in_0, in_1, in_2, in_3)
    mean_ptr, var_ptr, bias_ptr, weight_bn_ptr,
    # Output
    output_ptr,
    # Strides
    input_stride, weight_stride,
    output_stride,
    # Sizes
    N, C, H, W,  # N=1, C=512, H=64, W=64
    # BN parameters
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: Depthwise Conv + BatchNorm + ReLU
    Each program handles one output channel
    """
    # Get channel index
    pid = tl.program_id(0)
    
    if pid >= C:
        return
    
    # Calculate offsets
    # Depthwise conv: each channel processed independently
    # Weight shape: [512, 1, 7, 7], stride = 49 per channel
    weight_offset = pid * 49  # 7*7
    
    # Output offset for this channel
    out_offset_base = pid * H * W
    
    # Load batch norm parameters for this channel
    mean = tl.load(mean_ptr + pid)
    var = tl.load(var_ptr + pid)
    bn_weight = tl.load(weight_bn_ptr + pid)
    bn_bias = tl.load(bias_ptr + pid)
    
    # Compute standard deviation
    std = tl.sqrt(var + eps)
    
    # Compute normalization coefficients
    # batch_norm: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / std
    norm_weight = bn_weight * inv_std
    norm_bias = bn_bias - mean * bn_weight * inv_std
    
    # Compute depthwise conv + bn + relu for each position
    # Input: [N, C, H+6, W+6] = [1, 512, 70, 70] (padding implicitly handled)
    # Kernel: 7x7, output: 64x64
    # With stride=1, padding=0, we have: out_H = H_in - K + 1 = 70 - 7 + 1 = 64
    
    # Loop over output spatial positions
    for h in range(H):
        for w in range(W):
            # Compute conv for this output position
            # For depthwise: output[h,w] = sum over 7x7 kernel of input[h:h+7, w:w+7] * weight
            conv_sum = 0.0
            
            # Kernel loop - 7x7
            for kh in range(7):
                for kw in range(7):
                    # Input position
                    ih = h + kh
                    iw = w + kw
                    # Input index: [0, pid, ih, iw]
                    input_idx = pid * (70 * 70) + ih * 70 + iw
                    # Weight index: [pid, 0, kh, kw]
                    weight_idx = weight_offset + kh * 7 + kw
                    
                    # Load and multiply
                    inp_val = tl.load(input_ptr + input_idx)
                    w_val = tl.load(weight_ptr + weight_idx)
                    conv_sum += inp_val * w_val
            
            # Apply batch norm: (conv - mean) / std * weight + bias
            bn_out = conv_sum * norm_weight + norm_bias
            
            # Apply ReLU: max(0, x)
            relu_out = tl.maximum(bn_out, 0.0)
            
            # Store result
            out_idx = out_offset_base + h * W + w
            tl.store(output_ptr + out_idx, relu_out)


def fused_conv_bn_relu_impl(in_0, in_1, in_2, in_3, in_4, in_9):
    """
    Fused depthwise conv + batch norm + relu kernel
    in_0: running mean [512]
    in_1: running var [512]
    in_2: bias [512]
    in_3: weight [512]
    in_4: conv weight [512, 1, 7, 7]
    in_9: input [1, 512, 70, 70]
    """
    C = 512  # channels
    H = 64   # output height
    W = 64   # output width
    eps = 1e-05
    
    # Prepare input in contiguous format
    input_tensor = in_9.contiguous()
    weight_tensor = in_4.contiguous()
    
    # Create output tensor
    output = torch.empty((1, C, H, W), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Launch kernel - one program per channel
    grid = (C,)
    
    fused_conv_bn_relu_kernel[grid](
        input_tensor, weight_tensor,
        in_0, in_1, in_2, in_3,  # mean, var, bias, weight
        output,
        input_tensor.stride(0), weight_tensor.stride(0),
        output.stride(0),
        1, C, H, W,
        eps,
        BLOCK_SIZE=1024,
    )
    
    return output


# Wrapper function for FX graph replacement
@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3, in_4, in_9):
    return fused_conv_bn_relu_impl(in_0, in_1, in_2, in_3, in_4, in_9)


def replacement_func():
    """Return the kernel wrapper function"""
    return kernel_wrapper