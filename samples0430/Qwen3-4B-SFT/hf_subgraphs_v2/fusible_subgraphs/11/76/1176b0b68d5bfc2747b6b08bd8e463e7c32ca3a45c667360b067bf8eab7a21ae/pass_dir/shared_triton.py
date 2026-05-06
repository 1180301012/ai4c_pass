import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16, 'GROUP_M': 4}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 16, 'GROUP_M': 4}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['B', 'M', 'K', 'N'],
)
@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    a_stride_mb, a_stride_m, a_stride_k,
    b_stride_bb, b_stride_b, b_stride_k, b_stride_n,
    c_stride_bb, c_stride_m, c_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # program_id(0) = M*N-tile, program_id(1) = batch index
    pid_bn = tl.program_id(0)
    pid_b  = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id      = pid_bn // num_pid_in_group
    first_pid_m   = group_id * GROUP_M
    group_size_m  = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid_bn % num_pid_in_group) % group_size_m
    pid_n = (pid_bn % num_pid_in_group) // group_size_m

    # Load one-time base pointers per batch element
    # A: batch_start + m_offset * a_stride_m + k_offset * a_stride_k
    # B: (b * b_stride_bb) + k_offset * b_stride_k + n_offset * b_stride_n
    batch_mb = pid_b * b_stride_bb      # batch stride b (outermost, size B_outer)
    batch_b  = pid_b * b_stride_b       # batch stride for B's outer dim

    am_base = a_ptr + batch_mb          # A: no inner batch dim (always 0)
    bm_base = b_ptr + batch_b           # B: b-dimension offset

    # Initialise A and B with no k-column offset
    A = am_base + tl.arange(0, BLOCK_M)[:, None] * a_stride_m + tl.arange(0, BLOCK_K)[None, :] * a_stride_k
    B = bm_base + tl.arange(0, BLOCK_K)[:, None] * b_stride_k + tl.arange(0, BLOCK_N)[None, :] * b_stride_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k * BLOCK_K
        a = tl.load(
            A,
            mask=(tl.arange(0, BLOCK_M)[:, None] < M) & (k_off + tl.arange(0, BLOCK_K))[:, None] < K,
            other=0.0,
        )
        b = tl.load(
            B,
            mask=(k_off + tl.arange(0, BLOCK_K))[:, None] < K & (tl.arange(0, BLOCK_N)[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
        B = B + BLOCK_K   # advance k-column offset for next BLOCK_K step

    c_offs = tl.arange(0, BLOCK_M)[:, None] * c_stride_m + tl.arange(0, BLOCK_N)[None, :] * c_stride_n
    c_ptrs = c_ptr + pid_b * c_stride_bb + c_offs
    tl.store(
        c_ptrs,
        acc.to(a_ptr.dtype.element_ty),
        mask=(tl.arange(0, BLOCK_M)[:, None] < M) & (tl.arange(0, BLOCK_N)[None, :] < N),
    )


@torch.fx.wrap
def triton_batched_matmul(in_1, in_0):
    """
    Computes in_1 @ in_0.

    Handles both:
      - 2D case:  in_0.shape = (K, N)
      - 4D case:  in_0.shape = (B, K, D), in_1.shape = (B, M, K)
        Here B_batch = in_1.shape[0], M = in_1.shape[1], K = in_1.shape[2],
        D = in_0.shape[2], out = (B, M, D)

    Uses actual tensor strides for robustness.
    """
    out_dtype = in_1.dtype

    if in_0.dim() == 2:
        # 2-D: in_1 [M, K]  in_0 [K, N]  out [M, N], B = 1
        B = 1
        M = in_1.shape[0]
        K = in_1.shape[1]
        N = in_0.shape[1]
        a_stride_mb = 0                        # no batch dim
        a_stride_m  = in_1.stride(0)
        a_stride_k  = in_1.stride(1)
        b_stride_bb = in_0.stride(0)
        b_stride_b  = in_0.stride(1)
        b_stride_k  = in_0.stride(0)
        b_stride_n  = in_0.stride(1)
        c_stride_bb = out.stride(0)
        c_stride_m  = out.stride(1)
        c_stride_n  = out.stride(2)
    else:
        # 4-D: in_1 [B_outer, M, K]  in_0 [B_outer, K, D]
        B_outer = in_1.shape[0]
        M = in_1.shape[1]
        K = in_1.shape[2]
        N = in_0.shape[2]
        a_stride_mb = in_1.stride(0)          # outer-batch stride for A
        a_stride_m  = in_1.stride(1)
        a_stride_k  = in_1.stride(2)
        b_stride_bb = in_0.stride(0)          # outer-batch stride for B
        b_stride_b  = in_0.stride(1)
        b_stride_k  = in_0.stride(1)
        b_stride_n  = in_0.stride(2)
        c_stride_bb = out.stride(0)
        c_stride_m  = out.stride(1)
        c_stride_n  = out.stride(2)

    out = torch.empty((B_outer, M, N), dtype=out_dtype, device=in_1.device)

    def get_grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
            B_outer,
        )

    batched_matmul_kernel[get_grid](
        in_1, in_0, out,
        B_outer, M, N, K,
        a_stride_mb, a_stride_m, a_stride_k,
        b_stride_bb, b_stride_b, b_stride_k, b_stride_n,
        c_stride_bb, c_stride_m, c_stride_n,
    )

    return out