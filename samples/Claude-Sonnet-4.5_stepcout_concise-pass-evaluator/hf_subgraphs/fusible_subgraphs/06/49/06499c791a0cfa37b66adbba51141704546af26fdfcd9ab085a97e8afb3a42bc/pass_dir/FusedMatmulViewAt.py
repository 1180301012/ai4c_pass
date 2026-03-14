import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, shape0, shape1, shape2, shape3):
    """
    Pattern matching for @ operator followed by view.
    This matches the computation in yolo11n model.py exactly.
    shape0-shape3 are placeholders for the view dimensions.
    """
    tmp_0 = in_1 @ in_0
    tmp_1 = tmp_0.view(shape0, shape1, shape2, shape3)
    return tmp_1


def replacement_args(in_0, in_1, shape0, shape1, shape2, shape3):
    """
    Extract arguments needed for the replacement.
    """
    return (in_0, in_1, shape0, shape1, shape2, shape3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Optimized batched matmul kernel with better tiling.
    Computes C[b, m, n] = sum_k A[b, m, k] * B[b, k, n] for each batch b.
    """
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load A[b, m, k]
        a_ptrs = a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        # Load B[b, k, n]
        b_ptrs = b_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Compute
        accumulator += tl.dot(a, b)
    
    # Store result
    c = accumulator.to(tl.float32)
    
    c_ptrs = c_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


@torch.fx.wrap
def fused_matmul_view_at(in_0, in_1, shape0, shape1, shape2, shape3):
    """
    Fused implementation of @ + view using Triton.
    """
    # Get shapes
    original_shape_1 = in_1.shape
    
    # Flatten batch dimensions for easier processing
    in_1_flat = in_1.reshape(-1, in_1.shape[-2], in_1.shape[-1])
    in_0_flat = in_0.reshape(-1, in_0.shape[-2], in_0.shape[-1])
    
    batch_size = in_1_flat.shape[0]
    M = in_1_flat.shape[1]  # rows of in_1
    K = in_1_flat.shape[2]  # cols of in_1, rows of in_0
    N = in_0_flat.shape[2]  # cols of in_0
    
    # Allocate output
    out = torch.empty((batch_size, M, N), dtype=in_1.dtype, device=in_1.device)
    
    # Grid configuration (autotuning will pick best block sizes)
    grid = lambda META: (
        batch_size,
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel with autotuning
    batched_matmul_kernel[grid](
        in_1_flat, in_0_flat, out,
        M, N, K,
        in_1_flat.stride(0), in_1_flat.stride(1), in_1_flat.stride(2),
        in_0_flat.stride(0), in_0_flat.stride(1), in_0_flat.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
    )
    
    # Reshape to match the original batch structure
    batch_dims = original_shape_1[:-2]
    intermediate_shape = list(batch_dims) + [M, N]
    out = out.reshape(intermediate_shape)
    
    # Apply the view with the captured shape
    return out.view(shape0, shape1, shape2, shape3)


def replacement_func():
    return fused_matmul_view_at