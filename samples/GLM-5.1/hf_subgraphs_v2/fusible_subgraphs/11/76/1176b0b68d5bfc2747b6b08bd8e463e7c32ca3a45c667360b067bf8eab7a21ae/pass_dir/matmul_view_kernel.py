import torch
import triton
import triton.language as tl

# ============================================================
# Triton kernels for batched matmul
# ============================================================

# For N > 1: Standard tiled batched matmul using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_abatch,
    stride_bk, stride_bn, stride_bbatch,
    stride_cm, stride_cn, stride_cbatch,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program computes a [BLOCK_M, BLOCK_N] tile of the output for one batch element
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)

    # Create offset arrays for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to A and B tiles (starting position)
    a_ptrs = a_ptr + pid_batch * stride_abatch + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + pid_batch * stride_bbatch + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in float32 for numerical stability
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        # Masks for boundary handling
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=False)

        # Advance pointers for next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result
    c_ptrs = c_ptr + pid_batch * stride_cbatch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# For N = 1: Matrix-vector product (more efficient than tiled matmul for small N)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 256}, num_stages=4, num_warps=8),
    ],
    key=['M', 'K'],
)
@triton.jit
def matvec_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K,
    stride_am, stride_ak, stride_abatch,
    stride_bk, stride_bbatch,
    stride_cm, stride_cbatch,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in float32
    accumulator = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Main loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # Load B column vector (1D, shape BLOCK_K)
        b_ptrs = b_ptr + pid_batch * stride_bbatch + k_offs * stride_bk
        b_mask = k_offs < K
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Load A rows (2D, shape BLOCK_M x BLOCK_K)
        a_ptrs = a_ptr + pid_batch * stride_abatch + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Element-wise multiply and reduce along K for each row
        accumulator += tl.sum(a * b[None, :], axis=1)

    # Store result
    c_ptrs = c_ptr + pid_batch * stride_cbatch + offs_m * stride_cm
    c_mask = offs_m < M
    tl.store(c_ptrs, accumulator, mask=c_mask)


@torch.fx.wrap
def matmul_fused(in_0, in_1):
    """Fused batched matmul operation using Triton kernels."""
    # in_1 @ in_0: A is left operand (in_1), B is right operand (in_0)
    a = in_1  # [batch_dims..., M, K]
    b = in_0  # [batch_dims..., K, N]

    # Get shape info - use only allowed operations
    a_shape = a.size()
    b_shape = b.size()

    # Compute dimensions analytically
    # Flatten batch dimensions: a has shape [*batch_dims, M, K], b has shape [*batch_dims, K, N]
    ndim_a = a.dim()
    ndim_b = b.dim()
    
    # For contiguous tensors with shape [B1, B2, ..., M, K]:
    # stride for last dim (K) = 1
    # stride for second-to-last dim (M) = K
    # stride for batch dims depends on the specific layout
    # For contiguous layout: stride[i] = prod(shape[i+1:])
    
    M = a_shape[-2]
    K = a_shape[-1]
    N = b_shape[-1]
    
    # Compute batch size as product of all batch dims
    batch_size = 1
    for i in range(ndim_a - 2):
        batch_size *= a_shape[i]
    
    # Compute strides analytically for contiguous tensors
    # For a with shape [B1, B2, ..., M, K] (contiguous):
    # stride_abatch = M * K  (stride between consecutive batch elements when viewed as 3D)
    # stride_am = K  (stride along M dimension)
    # stride_ak = 1  (stride along K dimension)
    
    # But we need to handle the actual multi-dimensional strides
    # For a 4D contiguous tensor [B1, B2, M, K]:
    # strides are [B2*M*K, M*K, K, 1]
    # When flattened to [B1*B2, M, K], the "batch stride" is M*K
    
    stride_ak = a.stride(-1)  # = 1 for contiguous
    stride_am = a.stride(-2)  # = K for contiguous
    stride_abatch = a.stride(-3)  # = B2*M*K for 4D contiguous, or just M*K for 3D
    
    stride_bk = b.stride(-2)  # = N for contiguous
    stride_bn = b.stride(-1)  # = 1 for contiguous
    stride_bbatch = b.stride(-3)  # batch stride
    
    # For output: shape is [B1, B2, ..., M, N] (same batch dims as input)
    out_shape = list(a_shape[:-2]) + [M, N]
    out = torch.empty(out_shape, dtype=a.dtype, device=a.device)
    
    stride_cn = out.stride(-1)  # = 1 for contiguous
    stride_cm = out.stride(-2)  # = N for contiguous
    stride_cbatch = out.stride(-3)  # batch stride
    
    if N == 1:
        # Use matrix-vector kernel for N=1 case
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), batch_size)
        matvec_kernel[grid](
            a, b, out,
            M, K,
            stride_am, stride_ak, stride_abatch,
            stride_bk, stride_bbatch,
            stride_cm, stride_cbatch,
        )
    else:
        # Use standard batched matmul kernel for N > 1
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
            batch_size,
        )
        batched_matmul_kernel[grid](
            a, b, out,
            M, N, K,
            stride_am, stride_ak, stride_abatch,
            stride_bk, stride_bn, stride_bbatch,
            stride_cm, stride_cn, stride_cbatch,
        )

    return out