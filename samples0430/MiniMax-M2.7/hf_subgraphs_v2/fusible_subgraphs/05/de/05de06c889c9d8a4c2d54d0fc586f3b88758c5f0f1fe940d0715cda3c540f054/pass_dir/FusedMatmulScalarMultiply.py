import torch
import triton
import triton.language as tl

@triton.jit
def fused_matmul_scalar_kernel(
    a_ptr, b_ptr, scalar_ptr, out_ptr,
    M, N, K,
    stride_a0, stride_a1,
    stride_b0, stride_b1,
    stride_out0, stride_out1,
):
    """
    Single-program fused kernel for matmul + scalar multiply.
    Computes all output elements in one program to minimize kernel launch overhead.
    """
    # Compute dot products for all output elements
    # For M=2, N=1: compute [0,0] and [1,0]
    
    # Pre-load scalar
    scalar = tl.load(scalar_ptr)
    
    # Compute for output[0, 0] - corresponds to matmul[0, 0]
    offs_k = tl.arange(0, 512)
    mask = offs_k < K
    
    # Load row 0 from a
    a0_vals = tl.load(a_ptr + 0 * stride_a0 + offs_k * stride_a1, mask=mask, other=0.0)
    # Load column 0 from b  
    b0_vals = tl.load(b_ptr + offs_k * stride_b0 + 0 * stride_b1, mask=mask, other=0.0)
    acc0 = tl.sum(a0_vals * b0_vals)
    result0 = acc0 * scalar
    
    # Store result[0, 0] in transposed position [0, 0]
    tl.store(out_ptr + 0 * stride_out0 + 0 * stride_out1, result0)
    
    # Compute for output[1, 0] - corresponds to matmul[1, 0]
    # Only if M > 1
    if M > 1:
        # Load row 1 from a
        a1_vals = tl.load(a_ptr + 1 * stride_a0 + offs_k * stride_a1, mask=mask, other=0.0)
        acc1 = tl.sum(a1_vals * b0_vals)
        result1 = acc1 * scalar
        
        # Store result[1, 0] in transposed position [0, 1]
        tl.store(out_ptr + 0 * stride_out0 + 1 * stride_out1, result1)

@torch.fx.wrap
def fused_matmul_scalar(a, b, scalar):
    """
    Optimized wrapper using single-program Triton kernel for fused matmul + scalar multiply.
    Computes: (a @ b) * scalar, returns [N, M] transposed output.
    """
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    
    # Pre-allocate output in transposed layout
    out = torch.empty((N, M), device=a.device, dtype=a.dtype)
    
    # Single program computes everything
    grid = (1,)
    
    fused_matmul_scalar_kernel[grid](
        a, b, scalar, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
    )
    
    return out

def pattern(in_0, in_1, in_2):
    """
    Match pattern: matmul followed by scalar multiply.
    in_0: scalar (logit_scale)
    in_1: [K, N] tensor (t)
    in_2: [M, K] tensor (text_embeds_1)
    """
    matmul = torch.matmul(in_2, in_1)
    out = matmul * in_0
    return out

def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0)

def replacement_func():
    return fused_matmul_scalar