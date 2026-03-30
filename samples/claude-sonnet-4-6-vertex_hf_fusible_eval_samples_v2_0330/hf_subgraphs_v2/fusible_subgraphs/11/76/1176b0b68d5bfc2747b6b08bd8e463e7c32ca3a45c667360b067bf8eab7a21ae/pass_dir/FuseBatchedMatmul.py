import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    return torch.matmul(in_1, in_0)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        # Configs for large N (YOLO: M=64, K=400, N=400)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=5, num_warps=4),
        # Configs for small N (GCNet/S-ViPNAS: N=1)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 128}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=6, num_warps=2),
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Batched GEMM: C[b, :, :] = A[b, :, :] @ B[b, :, :]
    A: [batch, M, K], B: [batch, K, N], C: [batch, M, N]
    Grid: (pid_m, pid_n, pid_b)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Advance base pointers for this batch element
    a_ptr = a_ptr + pid_b * stride_ab
    b_ptr = b_ptr + pid_b * stride_bb
    c_ptr = c_ptr + pid_b * stride_cb

    # Row / col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initial pointers into A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Boundary masks for M and N (constant across K iterations)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulate in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Mask for the current K block (last block may be partial)
        mask_k = offs_k < (K - k * BLOCK_K)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        # Advance pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write output tile
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


@torch.fx.wrap
def triton_batched_matmul(in_0, in_1):
    """
    Compute in_1 @ in_0 for batched tensors.
    in_1: (..., M, K)
    in_0: (..., K, N)
    Returns: (..., M, N)
    """
    shape_1 = in_1.shape   # (..., M, K)
    shape_0 = in_0.shape   # (..., K, N)

    M = shape_1[-2]
    K = shape_1[-1]
    N = shape_0[-1]

    # Flatten all leading dims into a single batch dimension
    batch = 1
    for d in shape_1[:-2]:
        batch *= d

    # Make contiguous 3-D views
    a = in_1.reshape(batch, M, K)
    b = in_0.reshape(batch, K, N)
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    c = torch.empty((batch, M, N), dtype=in_1.dtype, device=in_1.device)

    grid = (
        triton.cdiv(M, 64),   # over-allocate; autotune adjusts BLOCK_M/N
        triton.cdiv(N, 64),
        batch,
    )
    # Use a lambda for the real autotuned grid
    def grid_fn(META):
        return (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
            batch,
        )

    batched_matmul_kernel[grid_fn](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
    )

    # Restore original leading dims
    out_shape = tuple(shape_1[:-2]) + (M, N)
    return c.reshape(out_shape)


def replacement_func():
    return triton_batched_matmul