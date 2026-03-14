import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """Match Conv2D + BatchNorm + Add pattern (ResNet block tail).
    
    This matches:
    1. Conv2D: conv2d(input, weight)
    2. BatchNorm: batch_norm(conv_out, mean, var, weight, bias)
    3. Add: bn_out + residual
    """
    # Conv2D operation
    conv_out = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    # BatchNorm operation
    bn_out = torch.nn.functional.batch_norm(conv_out, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    # Add residual
    bn_out += in_5
    return bn_out


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """Extract arguments for the fused kernel."""
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


# Fused Conv + BN + Add kernel
@triton.jit
def fused_conv_bn_add_kernel(
    # Input pointer
    input_ptr,
    # Conv weight pointer
    conv_weight_ptr,
    # BN parameters
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    # Residual pointer
    residual_ptr,
    # Output pointer
    output_ptr,
    # Shapes
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    # BN epsilon
    eps: tl.constexpr,
    # Strides
    stride_i_b: tl.constexpr,
    stride_i_c: tl.constexpr,
    stride_i_h: tl.constexpr,
    stride_i_w: tl.constexpr,
    stride_r_b: tl.constexpr,
    stride_r_c: tl.constexpr,
    stride_r_h: tl.constexpr,
    stride_r_w: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_c: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_w: tl.constexpr,
):
    """Fused Conv2D (1x1) + BatchNorm + Add kernel.
    
    This kernel performs:
    1. Conv2D with 1x1 kernel (pointwise convolution)
    2. BatchNorm using running statistics
    3. Element-wise addition with residual
    
    All in a single kernel to avoid memory traffic between operations.
    """
    # Get program ID - map to batch and channel
    pid = tl.program_id(0)
    pid_b = pid // C_out
    pid_c = pid % C_out
    
    # Bounds check
    if pid_b >= B:
        return
    
    # Precompute BN parameters for this channel
    mean = tl.load(mean_ptr + pid_c)
    var = tl.load(var_ptr + pid_c)
    weight = tl.load(weight_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)
    
    # Compute normalized weight and bias for fused conv+bn
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    # Fused: conv(x, weight / sqrt(var + eps)) + (bias - mean * weight / sqrt(var + eps))
    inv_std = 1.0 / tl.sqrt(var + eps)
    fused_weight = weight * inv_std
    fused_bias = bias - mean * weight * inv_std
    
    # Load conv weight for output channel
    # Conv weight shape: [C_out, C_in, 1, 1]
    # Stored as [C_out, C_in] in row-major
    conv_weight = tl.load(conv_weight_ptr + pid_c * C_in + tl.arange(0, C_in))
    
    # Process all spatial positions for this batch and channel
    for h in range(H):
        for w in range(W):
            # Compute input offset: [B, C_in, H, W]
            # input[pid_b, :, h, w] starts at pid_b * stride_i_b + h * stride_i_h + w * stride_i_w
            input_base = pid_b * stride_i_b + h * stride_i_h + w * stride_i_w
            input_vals = tl.load(input_ptr + input_base + tl.arange(0, C_in) * stride_i_c)
            
            # Convolution: dot product of input channels with conv weight
            conv_out = tl.sum(input_vals * conv_weight)
            
            # Apply BN (fused)
            bn_out = conv_out * fused_weight + fused_bias
            
            # Load residual and add
            residual_base = pid_b * stride_r_b + pid_c * stride_r_c + h * stride_r_h + w * stride_r_w
            residual = tl.load(residual_ptr + residual_base)
            output = bn_out + residual
            
            # Store output
            output_base = pid_b * stride_o_b + pid_c * stride_o_c + h * stride_o_h + w * stride_o_w
            tl.store(output_ptr + output_base, output)


@torch.fx.wrap
def fused_conv_bn_add_kernel_wrapper(
    mean, var, weight, bias, conv_weight, residual, input, eps=1e-05
):
    """Wrapper for the fused Conv + BN + Add kernel.
    
    This fuses:
    1. Conv2D (1x1) - pointwise conv across channels
    2. BatchNorm - using running mean/var
    3. Add - residual connection
    
    Into a single GPU kernel for maximum performance.
    """
    B, C_in, H, W = input.shape
    C_out = conv_weight.shape[0]
    
    # Allocate output
    output = torch.empty_like(residual)
    
    # Total programs = B * C_out (each program handles one batch-channel combination)
    num_programs = B * C_out
    
    # Launch kernel
    grid = (num_programs,)
    
    # Get strides
    input = input.contiguous()
    residual = residual.contiguous()
    output = output.contiguous()
    
    fused_conv_bn_add_kernel[grid](
        input,
        conv_weight,
        mean,
        var,
        weight,
        bias,
        residual,
        output,
        B,
        C_in,
        C_out,
        H,
        W,
        eps,
        # Input strides
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        # Residual strides
        residual.stride(0), residual.stride(1), residual.stride(2), residual.stride(3),
        # Output strides
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    return output


def replacement_func():
    """Return the fused kernel function."""
    return fused_conv_bn_add_kernel_wrapper