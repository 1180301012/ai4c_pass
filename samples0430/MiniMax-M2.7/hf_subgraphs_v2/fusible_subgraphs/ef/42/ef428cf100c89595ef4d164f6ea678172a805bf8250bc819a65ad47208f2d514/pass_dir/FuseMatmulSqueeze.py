import torch
import triton
import triton.language as tl


@triton.jit
def matmul_squeeze_kernel(
    out_ptr,
    a_ptr, b_ptr,
    K: tl.constexpr, N: tl.constexpr,
    stride_a2: tl.constexpr,
    stride_b1: tl.constexpr, stride_b2: tl.constexpr,
    stride_out1: tl.constexpr,
):
    """
    Fused matmul + squeeze(1) kernel.
    
    Input a: [1, 1, K]
    Input b: [1, K, N]
    Output: [1, N] (squeezed from [1, 1, N])
    
    Simple and efficient dot product per thread.
    """
    pid = tl.program_id(0)
    
    if pid >= N:
        return
    
    # Compute dot product for output element pid
    acc = 0.0
    for k in range(K):
        a_val = tl.load(a_ptr + k * stride_a2)
        b_val = tl.load(b_ptr + k * stride_b1 + pid * stride_b2)
        acc += a_val * b_val
    
    tl.store(out_ptr + pid * stride_out1, acc)


@torch.fx.wrap
def fused_matmul_squeeze_wrapper(a, b):
    """
    Wrapper function for the fused matmul + squeeze operation.
    
    Args:
        a: Tensor of shape [1, 1, K] (bfloat16 or float16)
        b: Tensor of shape [1, K, N] (bfloat16 or float16)
    
    Returns:
        Tensor of shape [1, N] (squeezed from [1, 1, N])
    """
    K = a.shape[2]  # = 249
    N = b.shape[2]  # = 64
    
    # Allocate output tensor
    out = torch.empty([1, N], dtype=a.dtype, device=a.device)
    
    # Strides
    stride_a2 = a.stride(2)
    stride_b1 = b.stride(1)
    stride_b2 = b.stride(2)
    stride_out1 = out.stride(1)
    
    # Launch kernel
    matmul_squeeze_kernel[(N,)](
        out,
        a, b,
        K, N,
        stride_a2,
        stride_b1, stride_b2,
        stride_out1,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match the pattern: matmul followed by squeeze(1)
    
    Original computation:
        matmul = torch.matmul(in_0, in_1)  # [1, 1, 249] @ [1, 249, 64] -> [1, 1, 64]
        tmp_1 = matmul.squeeze(1)          # [1, 1, 64] -> [1, 64]
    """
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return tmp_1


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the replacement.
    """
    return (in_0, in_1)


def replacement_func():
    """
    Return the fused kernel wrapper function.
    """
    return fused_matmul_squeeze_wrapper