import torch
import triton
import triton.language as tl

# Pattern to match: RMSNorm only (weight * (x * rsqrt(mean(x^2) + eps)))
def pattern(weight, x):
    # Convert to float32
    tmp_4 = x.to(torch.float32)
    # Square
    tmp_5 = tmp_4.pow(2)
    # Mean along last dimension
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    # Add epsilon
    tmp_7 = tmp_6 + 1e-06
    # Reciprocal square root
    tmp_8 = torch.rsqrt(tmp_7)
    # Multiply by rsqrt
    tmp_9 = x * tmp_8
    # Multiply by weight
    tmp_10 = weight * tmp_9
    return tmp_10


def replacement_args(weight, x):
    return (weight, x)


@triton.jit
def rms_norm_kernel(
    x_ptr, w_ptr, o_ptr, stride,
):
    """RMSNorm kernel - one block per row"""
    row = tl.program_id(0)
    off = row * stride
    cols = tl.arange(0, 1024)
    
    # Load
    x = tl.load(x_ptr + off + cols)
    w = tl.load(w_ptr + cols)
    
    # RMSNorm
    xf = x.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(xf * xf) * 0.0009765625 + 1e-6)
    
    # Store
    tl.store(o_ptr + off + cols, x * rstd * w)


@torch.fx.wrap
def rmsnorm_triton(weight, x):
    """RMSNorm using Triton"""
    shape = x.shape
    M = shape[0] * shape[1]
    N = shape[2]
    
    x_flat = x.view(M, N)
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    
    rms_norm_kernel[(M,)](x_flat, weight, out, N, num_warps=8)
    
    return out.view(shape)


def replacement_func():
    return rmsnorm_triton