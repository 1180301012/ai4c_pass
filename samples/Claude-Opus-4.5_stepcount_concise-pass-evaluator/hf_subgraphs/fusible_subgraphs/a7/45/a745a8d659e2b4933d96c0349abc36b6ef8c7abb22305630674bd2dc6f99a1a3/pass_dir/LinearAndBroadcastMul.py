import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern for rtmpose-l: independent linear and broadcast multiply
    in_0: weight [out_features, in_features] for linear
    in_1: scale vector [out_features] for multiply
    in_2: input tensor [batch, seq, out_features] for multiply
    in_3: input tensor [batch, seq, in_features] for linear
    """
    tmp_2 = torch.nn.functional.linear(in_3, in_0, None)
    tmp_3 = in_2 * in_1
    return (tmp_3, tmp_2)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def broadcast_mul_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    N,  # Total elements
    D,  # Last dimension (scale dimension)
    BLOCK_SIZE: tl.constexpr,
):
    """Broadcast multiply: x * scale where scale is broadcast along last dim"""
    pid = tl.program_id(0)
    
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Compute which scale element to use
    scale_idx = offs % D
    
    # Load x and scale
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + scale_idx, mask=mask, other=0.0)
    
    # Multiply
    out = x * scale
    
    # Store
    tl.store(out_ptr + offs, out, mask=mask)


@torch.fx.wrap
def linear_and_broadcast_mul(weight, scale, x_mul, x_linear):
    """
    Combined linear and broadcast multiply operations
    weight: [out_features, in_features]
    scale: [out_features]
    x_mul: [batch, seq, out_features]
    x_linear: [batch, seq, in_features]
    """
    # Perform linear operation using PyTorch (cuBLAS is highly optimized for this)
    linear_out = torch.nn.functional.linear(x_linear, weight, None)
    
    # Perform optimized broadcast multiply using Triton
    N = x_mul.numel()
    D = scale.shape[0]
    
    # Ensure contiguous
    x_mul_cont = x_mul.contiguous()
    scale_cont = scale.contiguous()
    
    # Output tensor
    mul_out = torch.empty_like(x_mul_cont)
    
    # Grid
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    
    # Launch kernel
    broadcast_mul_kernel[grid](
        x_mul_cont,
        scale_cont,
        mul_out,
        N,
        D,
    )
    
    # Reshape to original shape if needed
    mul_out = mul_out.view_as(x_mul)
    
    return (mul_out, linear_out)


def replacement_func():
    return linear_and_broadcast_mul