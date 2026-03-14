import torch
import triton
import triton.language as tl


def pattern(in_3, tmp_1):
    """Match transpose followed by multiply (in original order)."""
    tmp_2 = tmp_1.transpose(-1, -2)
    # Match in same order as model: in_3 * tmp_2
    return in_3 * tmp_2


def replacement_args(in_3, tmp_1):
    return (in_3, tmp_1)


# Kernel for transpose + mul fused
@triton.jit
def transpose_mul_kernel(
    a_ptr, b_ptr, out_ptr,
    N: tl.constexpr, K: tl.constexpr,
    stride_a_0, stride_a_1, stride_a_2, stride_a_3,
    stride_b_0, stride_b_1, stride_b_2, stride_b_3,
    stride_out_0, stride_out_1, stride_out_2, stride_out_3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused transpose + multiply:
    Input a: [..., N, K] (needs transpose to [..., K, N])
    Input b: [..., K, N] (after transpose, this matches)
    Output: [..., K, N]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N * K
    
    # Original indices in a (before transpose): a[..., n, k]
    # After transpose: a_transposed[..., k, n]
    # So for position (k, n) in output:
    # - we need a[..., n, k] which is at position n*K + k in a
    # - we need b[..., k, n] which is at position k*N + n in b
    
    n_idx = offs % N
    k_idx = offs // N
    
    mask_n = n_idx < N
    mask_k = k_idx < K
    valid_mask = mask_n & mask_k
    
    # Load from a - need to transpose (swap n and k)
    a_offset = (n_idx * K + k_idx)  # index in a before transpose
    a_offset = a_offset * stride_a_3  # simplify - just use 1D view
    a_val = tl.load(a_ptr + a_offset, mask=valid_mask, other=0.0)
    
    # Load from b (already in correct layout)
    b_offset = (k_idx * stride_b_2 + n_idx * stride_b_3)
    b_val = tl.load(b_ptr + b_offset, mask=valid_mask, other=0.0)
    
    # Multiply
    out_val = a_val * b_val
    
    # Store in output
    out_offset = (k_idx * stride_out_2 + n_idx * stride_out_3)
    tl.store(out_ptr + out_offset, out_val, mask=valid_mask)


@torch.fx.wrap
def transpose_mul_kernel_wrapper(a, b):
    # Handle the case where shapes might be 4D
    a_shape = a.shape
    b_shape = b.shape
    
    # For 4D: [..., N, K] and [..., K, N]
    # Output: [..., K, N]
    N = a_shape[-1]
    K = a_shape[-2]
    
    # Create output with same shape as b
    output = torch.empty_like(b)
    
    BLOCK_SIZE = 4096
    n_elements = N * K
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Flatten for kernel
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    out_flat = output.reshape(-1)
    
    transpose_mul_kernel[(num_programs,)](
        a_ptr=a_flat,
        b_ptr=b_flat,
        out_ptr=out_flat,
        N=N,
        K=K,
        stride_a_0=a.stride(0), stride_a_1=a.stride(1), stride_a_2=a.stride(2), stride_a_3=a.stride(3),
        stride_b_0=b.stride(0), stride_b_1=b.stride(1), stride_b_2=b.stride(2), stride_b_3=b.stride(3),
        stride_out_0=output.stride(0), stride_out_1=output.stride(1), stride_out_2=output.stride(2), stride_out_3=output.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return transpose_mul_kernel_wrapper