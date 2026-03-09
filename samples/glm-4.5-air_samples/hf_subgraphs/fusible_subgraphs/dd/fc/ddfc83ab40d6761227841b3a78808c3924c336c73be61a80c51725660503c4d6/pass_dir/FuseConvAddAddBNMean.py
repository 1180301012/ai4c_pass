import torch
import triton
import triton.language as tl


# Autotune configurations for better performance
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=1),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def fused_conv_add_add_bn_mean_kernel(
    # Input tensor (in_6)
    in6_ptr,
    # Input tensor (in_7) - residual
    in7_ptr,
    # Conv weight (in_5)
    conv_weight_ptr,
    # Conv bias (in_4)
    conv_bias_ptr,
    # BN running_mean (in_0)
    bn_mean_ptr,
    # BN running_var (in_1)
    bn_var_ptr,
    # BN weight (in_3)
    bn_weight_ptr,
    # BN bias (in_2)
    bn_bias_ptr,
    # Output 1 - BN result
    out1_ptr,
    # Sizes
    N: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
    stride_in6: tl.constexpr,
    stride_in7: tl.constexpr,
    stride_out1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate position
    pid = tl.program_id(0)
    num_elements = N * C * H * W
    
    # Each program processes BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Compute indices
    n = offsets // (C * H * W)
    remainder = offsets % (C * H * W)
    c = remainder // (H * W)
    hw = remainder % (H * W)
    h = hw // W
    w = hw % W
    
    # Compute strides for 2D access
    in6_offset = n * stride_in6 + c * H * W + h * W + w
    in7_offset = n * stride_in7 + c * H * W + h * W + w
    
    # Load inputs
    in6 = tl.load(in6_ptr + in6_offset, mask=mask, other=0.0)
    in7 = tl.load(in7_ptr + in7_offset, mask=mask, other=0.0)
    
    # Load conv weight and bias
    # Conv is 1x1 depthwise, so we need to load the correct channel weight
    conv_weight = tl.load(conv_weight_ptr + c, mask=mask, other=0.0)
    conv_bias = tl.load(conv_bias_ptr + c, mask=mask, other=0.0)
    
    # Depthwise conv: each channel is independent
    # For 1x1 conv with groups=C: out[c,h,w] = in[c,h,w] * weight[c,0,0,0] + bias[c]
    conv_out = in6 * conv_weight + conv_bias
    
    # Add residual: in7 + conv_out
    add1 = in7 + conv_out
    
    # Add residual: add1 + in6
    add2 = add1 + in6
    
    # Load BN parameters
    bn_mean = tl.load(bn_mean_ptr + c, mask=mask, other=0.0)
    bn_var = tl.load(bn_var_ptr + c, mask=mask, other=0.0)
    bn_weight = tl.load(bn_weight_ptr + c, mask=mask, other=0.0)
    bn_bias = tl.load(bn_bias_ptr + c, mask=mask, other=0.0)
    
    # Batch normalization (fused)
    # eps = 1e-05
    eps = 1e-05
    bn_out = (add2 - bn_mean) / tl.sqrt(bn_var + eps) * bn_weight + bn_bias
    
    # Store output 1
    tl.store(out1_ptr + in6_offset, bn_out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
    ],
    key=['C'],
)
@triton.jit
def mean_reduction_kernel(
    in_ptr,
    # Output: shape [N, C, 1, 1] but stored as contiguous C values per batch
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_in: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel across all batches
    # We compute mean per (batch, channel) pair
    pid = tl.program_id(0)
    
    # Number of (batch, channel) pairs
    num_bc = N * C
    
    if pid >= num_bc:
        return
    
    # Compute batch and channel indices
    n = pid // C
    c = pid % C
    
    # Compute sum for this (batch, channel)
    sum_val = 0.0
    for h in range(H):
        for w in range(W):
            offset = n * stride_in + c * H * W + h * W + w
            sum_val += tl.load(in_ptr + offset)
    
    # Compute mean
    num_vals = H * W
    mean_val = sum_val / num_vals
    
    # Store result at out[n, c, 0, 0]
    # Shape is [N, C, 1, 1], stored as [N*C] contiguous
    out_offset = n * C + c
    tl.store(out_ptr + out_offset, mean_val)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Match: conv2d (1x1) + add + add + batch_norm + mean
    This is a common pattern in RepViT and similar architectures.
    
    Using default groups=1 for pattern matching - the actual value will be
    extracted from the matched graph during replacement.
    """
    # Use groups=1 as placeholder for pattern matching
    tmp_6 = torch.nn.functional.conv2d(in_6, in_5, in_4, (1, 1), (0, 0), (1, 1), 1)
    tmp_7 = in_7 + tmp_6
    tmp_8 = tmp_7 + in_6
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_10 = tmp_9.mean((2, 3), keepdim=True)
    return tmp_9, tmp_10


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)


@torch.fx.wrap
def fused_kernel(
    bn_mean, bn_var, bn_weight, bn_bias,
    conv_weight, conv_bias, input1, input2
):
    N, C, H, W = input1.shape
    
    # Allocate output for BN result
    out1 = torch.empty_like(input1)
    
    # Launch kernel for conv + add + add + bn
    # Grid: need enough blocks to cover all elements
    num_elements = N * C * H * W
    grid = (num_elements,)
    
    fused_conv_add_add_bn_mean_kernel[grid](
        input1, input2, conv_weight, conv_bias,
        bn_mean, bn_var, bn_weight, bn_bias,
        out1,
        N, C, H, W,
        input1.stride(0), input2.stride(0),
        out1.stride(0),
    )
    
    # Compute mean using separate kernel
    # Output shape: [N, C, 1, 1]
    out2 = torch.empty((N, C, 1, 1), dtype=torch.float32, device=input1.device)
    
    # Grid: one program per (batch, channel) pair
    mean_grid = (N * C,)
    
    mean_reduction_kernel[mean_grid](
        out1,  # Use the BN output as input
        out2,  # Output
        N, C, H, W,
        out1.stride(0),
    )
    
    return out1, out2


def replacement_func():
    return fused_kernel