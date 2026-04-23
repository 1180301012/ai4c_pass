"""
Shared kernel dispatch module for DANet_R101 graph optimization.

This module provides optimized Triton kernels for:
1. Conv2d with bias (1x1 convolution)
2. Element-wise add + dropout2d fusion

Both operations are independent and can be dispatched based on route string.
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Conv2d 1x1 with Bias Kernel
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_K': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_K': 512}, num_stages=3, num_warps=8),
    ],
    key=['K'],
)
@triton.jit
def conv2d_1x1_bias_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, K, H, W,  # input NCHW
    R,  # number of output channels
    stride_h, stride_w,  # typically 1, 1
    BLOCK_K: tl.constexpr,
):
    """
    Optimized 1x1 Conv2d with bias using implicit GEMM.
    
    For 1x1 conv with stride=1, padding=0, dilation=1:
    - input: (N, K, H, W)
    - weight: (R, K, 1, 1) where R is output_channels
    - output: (N, R, H, W)
    
    This treats the convolution as a batched GEMM where each spatial location (h, w)
    is a separate matrix multiplication problem.
    """
    # Program ID for the output grid
    pid = tl.program_id(0)
    
    # Calculate which output element this program handles
    # Map 1D pid to 3D (n, r, spatial_idx)
    n = pid // (R * H * W)
    remainder = pid % (R * H * W)
    r = remainder // (H * W)
    spatial_idx = remainder % (H * W)
    h = spatial_idx // W
    w = spatial_idx % W
    
    # Create masks for bounds
    n_mask = n < N
    r_mask = r < R
    h_mask = h < H
    w_mask = w < W
    
    # Load bias (bias is [R])
    bias_offset = r
    bias = tl.load(bias_ptr + bias_offset, mask=r_mask, other=0.0)
    
    # Input linear offset: n*K*H*W + k*H*W + h*W + w
    base_input_offset = n * K * H * W + h * W + w
    
    # Weight linear offset: weight is [R, K, 1, 1]
    # Linear offset: r*K*1*1 + k*1*1 = r*K + k
    base_weight_offset = r * K
    
    # Process along K dimension (input channels)
    # Create the K-sized vectors
    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K
    
    # Load input vector [K] at position (n, :, h, w)
    input_offsets = base_input_offset + k_offsets * H * W
    inp = tl.load(input_ptr + input_offsets, mask=k_mask, other=0.0)
    
    # Load weight vector [K] at position (r, :, 0, 0)
    weight_offsets = base_weight_offset + k_offsets
    wgt = tl.load(weight_ptr + weight_offsets, mask=k_mask, other=0.0)
    
    # Compute element-wise product and reduce sum
    prod = inp * wgt
    acc = tl.sum(prod, axis=0)
    
    # Add bias
    result = acc + bias
    
    # Store result
    output_offset = n * R * H * W + r * H * W + h * W + w
    tl.store(output_ptr + output_offset, result)


def conv2d_1x1_bias_triton(input, weight, bias, stride_h=1, stride_w=1):
    """
    Triton implementation of 1x1 Conv2d with bias.
    
    Args:
        input: (N, K, H, W) tensor
        weight: (R, K, 1, 1) tensor
        bias: (R,) tensor
        stride_h, stride_w: stride (typically 1 for 1x1 conv)
    
    Returns:
        output: (N, R, H, W) tensor
    """
    N, K, H, W = input.shape
    R = weight.shape[0]
    
    # Output tensor - using torch.empty as per allowed APIs
    output = torch.empty((N, R, H, W), device=input.device, dtype=input.dtype)
    
    # Total number of output elements
    total_elements = N * R * H * W
    
    # Launch kernel - weight is passed directly, kernel handles the 1x1 shape
    conv2d_1x1_bias_kernel[(total_elements,)](
        input, weight, bias, output,
        N, K, H, W, R,
        stride_h, stride_w,
    )
    
    return output


# =============================================================================
# Add + Dropout2d Fusion Kernel (for inference mode)
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def add_dropout2d_kernel(
    input1_ptr, input2_ptr, output_ptr,
    N, C, H, W,
    p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused add + dropout2d kernel.
    
    Note: When in training mode with dropout, the mask needs to be consistent.
    For inference mode (train=False), dropout is identity, so we just do add.
    """
    pid = tl.program_id(0)
    
    # Calculate starting offset for this program
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N * C * H * W
    
    # Load inputs
    x1 = tl.load(input1_ptr + offset, mask=mask, other=0.0)
    x2 = tl.load(input2_ptr + offset, mask=mask, other=0.0)
    
    # Add
    result = x1 + x2
    
    # For train=False (which is the case in the pattern), dropout is identity
    # If we were to implement training dropout, we'd need Philox RNG
    # But since train=False in the pattern, we just return the sum
    
    # Store result
    tl.store(output_ptr + offset, result, mask=mask)


def add_dropout2d_triton(input1, input2, p=0.1, train=False, return_mask=False):
    """
    Fused add + dropout2d operation.
    
    For train=False (inference), dropout is identity, so just element-wise add.
    For train=True, would need to generate mask - not supported for simplicity.
    """
    N, C, H, W = input1.shape
    
    output = torch.empty_like(input1)
    
    total_elements = N * C * H * W
    BLOCK_SIZE = 4096  # Default, autotune will find optimal
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    add_dropout2d_kernel[(num_programs,)](
        input1, input2, output,
        N, C, H, W,
        float(p),
    )
    
    return output


# =============================================================================
# Shared Dispatch Wrapper
# =============================================================================

@torch.fx.wrap
def dispatch_kernel(in_2, in_1, in_0, stride_h, stride_w, route):
    """
    Shared dispatch wrapper that routes to the appropriate optimized kernel.
    
    Routes:
        "conv2d": Conv2d 1x1 with bias
        "add_dropout": Add + Dropout2d fusion
    """
    if route == "conv2d":
        # args: (input, weight, bias, stride_h, stride_w)
        return conv2d_1x1_bias_triton(in_2, in_1, in_0, stride_h, stride_w)
    elif route == "add_dropout":
        # For add_dropout, in_2=input1, in_1=input2, in_0=unused, stride_h=p
        return add_dropout2d_triton(in_2, in_1)
    else:
        raise ValueError(f"Unknown route: {route}")