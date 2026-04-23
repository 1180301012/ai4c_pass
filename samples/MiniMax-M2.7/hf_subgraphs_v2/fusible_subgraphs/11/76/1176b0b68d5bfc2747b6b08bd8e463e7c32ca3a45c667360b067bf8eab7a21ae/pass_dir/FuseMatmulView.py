import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel_4d(
    a_ptr, b_ptr, c_ptr,
    B, G, K, W, H,  # Full shapes
    stride_a0, stride_a1, stride_a2, stride_a3,
    stride_b0, stride_b1, stride_b2, stride_b3,
    stride_c0, stride_c1, stride_c2, stride_c3,
    BLOCK_SIZE: tl.constexpr
):
    """
    Batched matrix multiply kernel for 4D tensors.
    Computes (B, G, K, W) @ (B, G, H, W) -> (B, G, K, H)
    
    The matmul is: result[k, h] = sum_w A[k, w] * B[h, w]
    Which is: sum over w of outer products of rows of A and rows of B
    """
    b_idx = tl.program_id(0)
    g_idx = tl.program_id(1)
    
    offs_k = tl.arange(0, BLOCK_SIZE)
    offs_h = tl.arange(0, BLOCK_SIZE)
    
    # Result accumulator (K, H)
    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Iterate over W dimension
    for w in range(W):
        # Load A[:, w] - a column of A with shape (K,)
        # a_ptr offsets: (b_idx, g_idx, k, w)
        a_ptrs = (a_ptr + 
                  stride_a0 * b_idx + stride_a1 * g_idx +
                  stride_a2 * offs_k + 
                  stride_a3 * w)
        a_mask = offs_k < K
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # shape (K,)
        
        # Load B[:, w] - a column of B with shape (H,)
        b_ptrs = (b_ptr + 
                  stride_b0 * b_idx + stride_b1 * g_idx +
                  stride_b2 * offs_h + 
                  stride_b3 * w)
        b_mask = offs_h < H
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # shape (H,)
        
        # Outer product: a[:, None] * b[None, :] = (K, 1) * (1, H) = (K, H)
        # This is equivalent to a column-vector times a row-vector
        accumulator += a[:, None] * b[None, :]
    
    # Store result: (B, G, K, H)
    c_ptrs = (c_ptr + 
              stride_c0 * b_idx + stride_c1 * g_idx +
              stride_c2 * offs_k[:, None] +
              stride_c3 * offs_h[None, :])
    c_mask = (offs_k[:, None] < K) & (offs_h[None, :] < H)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)


def pattern(in_0, in_1):
    """
    Match matmul pattern using @ operator.
    """
    matmul_result = in_1 @ in_0
    return matmul_result


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@torch.fx.wrap
def matmul_wrapper(in_0, in_1):
    """
    Triton-based batched matmul for 4D tensors.
    Returns the matmul result - the view operation will be applied by the graph.
    """
    B, G, K, W = in_1.shape
    _, _, H, _ = in_0.shape
    
    # Output: (B, G, K, H)
    c = torch.empty((B, G, K, H), device=in_1.device, dtype=in_1.dtype)
    
    grid = (B, G)
    BLOCK_SIZE = 64
    
    matmul_kernel_4d[grid](
        in_1, in_0, c,
        B, G, K, W, H,
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        BLOCK_SIZE
    )
    
    return c


def replacement_func():
    return matmul_wrapper