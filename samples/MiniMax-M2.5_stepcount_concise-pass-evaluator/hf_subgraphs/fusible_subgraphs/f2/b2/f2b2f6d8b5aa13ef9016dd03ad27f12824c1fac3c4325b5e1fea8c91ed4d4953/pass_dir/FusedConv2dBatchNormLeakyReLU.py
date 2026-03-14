import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match Conv2d + BatchNorm + LeakyReLU pattern
    in_0: running_mean
    in_1: running_var  
    in_2: bias
    in_3: weight
    in_4: conv weight
    in_5: input tensor
    """
    # Conv2d with stride=(1,1), padding=(1,1), dilation=(1,1), groups=1
    tmp_6 = torch.conv2d(in_5, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    # BatchNorm with eps=1e-05, momentum=0.1
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    # LeakyReLU with negative_slope=0.01, inplace=True
    tmp_8 = torch.nn.functional.leaky_relu(tmp_7, 0.01, True)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


# Autotune configurations for different input sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def fused_bn_relu_kernel(
    # Input pointers - conv output
    input_ptr,
    # BN parameters
    running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    # Output pointer
    output_ptr,
    # Tensor dimensions: N = total pixels (B*H*W), C = channels
    N, C,
    # Strides
    stride_n, stride_c,
    # Meta parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused BatchNorm + LeakyReLU kernel.
    This kernel applies:
    1. BatchNorm: (x - mean) * weight / sqrt(var + eps) + bias
    2. LeakyReLU: x > 0 ? x : x * negative_slope
    
    We process in a channel-parallel fashion: each program handles 
    BLOCK_SIZE_N channels for BLOCK_SIZE_M pixels.
    """
    # Get program ids
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(C, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, N - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = pid // num_pid_in_group % num_pid_n
    
    # Calculate offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Mask for bounds
    mask_m = offs_m < N
    mask_n = offs_n < C
    
    # Load input data: shape [BLOCK_SIZE_M, BLOCK_SIZE_N]
    input_ptrs = input_ptr + offs_m[:, None] * stride_n + offs_n[None, :] * stride_c
    inp = tl.load(input_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Load BN parameters for these channels
    rm = tl.load(running_mean_ptr + offs_n, mask=mask_n, other=0.0)
    rv = tl.load(running_var_ptr + offs_n, mask=mask_n, other=1.0)
    bn_w = tl.load(bn_weight_ptr + offs_n, mask=mask_n, other=1.0)
    bn_b = tl.load(bn_bias_ptr + offs_n, mask=mask_n, other=0.0)
    
    # Apply batch norm: (x - mean) * weight / sqrt(var + eps) + bias
    eps = 1e-5
    normalized = (inp - rm) * bn_w / tl.sqrt(rv + eps) + bn_b
    
    # Apply LeakyReLU: x > 0 ? x : x * negative_slope
    negative_slope = 0.01
    result = tl.where(normalized > 0, normalized, normalized * negative_slope)
    
    # Store result
    output_ptrs = output_ptr + offs_m[:, None] * stride_n + offs_n[None, :] * stride_c
    tl.store(output_ptrs, result, mask=mask_m[:, None] & mask_n[None, :])


@torch.fx.wrap
def fused_conv_bn_relu_wrapper(
    running_mean, running_var, bn_bias, bn_weight, 
    conv_weight, input_tensor
):
    """
    Wrapper function that orchestrates the fused Conv + BN + ReLU operation.
    
    Approach:
    1. Use cuDNN (via torch.nn.functional.conv2d) for the convolution part
    2. Use a custom Triton kernel for fused BN + LeakyReLU
    
    This fuses 2 operations (BN + ReLU) into 1 kernel, while leveraging
    cuDNN's highly optimized convolution implementation.
    """
    # Move inputs to GPU if needed
    if not input_tensor.is_cuda:
        input_tensor = input_tensor.cuda()
    if not conv_weight.is_cuda:
        conv_weight = conv_weight.cuda()
    if not running_mean.is_cuda:
        running_mean = running_mean.cuda()
    if not running_var.is_cuda:
        running_var = running_var.cuda()
    if not bn_weight.is_cuda:
        bn_weight = bn_weight.cuda()
    if not bn_bias.is_cuda:
        bn_bias = bn_bias.cuda()
    
    # First do convolution using PyTorch's optimized implementation (cuDNN)
    conv_output = torch.nn.functional.conv2d(
        input_tensor, conv_weight, None, (1, 1), (1, 1), (1, 1), 1
    )
    
    # Now apply fused BN + LeakyReLU using Triton kernel
    # This fuses two operations (BN and ReLU) into a single kernel
    B, C, H, W = conv_output.shape
    N = B * H * W  # Total number of spatial positions
    
    # Allocate output
    output = torch.empty_like(conv_output)
    
    # Calculate grid - we want one program per pixel group
    grid = (triton.cdiv(N, 1) * triton.cdiv(C, 1024),)
    
    # Launch kernel
    fused_bn_relu_kernel[grid](
        conv_output,
        running_mean, running_var, bn_weight, bn_bias,
        output,
        N, C,
        conv_output.stride(0), conv_output.stride(1),
    )
    
    return output


def replacement_func():
    return fused_conv_bn_relu_wrapper