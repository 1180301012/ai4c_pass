import torch
import triton
import triton.language as tl

# ===========================================================================
# MatmulFusion: replace the batched GEMM  in_1 @ in_0  with a Triton kernel.
#
# For small non-power-of-2 K values (K=16,19,27,32,40,64) cuBLAS incurs
# padding overhead; Triton can match or beat it while also giving the FX
# compiled graph one fewer Python dispatch overhead bubble.
# ===========================================================================


# ---------------------------------------------------------------------------
# Batched GEMM kernel
#   C[bh, m, n] = sum_k  A[bh, m, k] * B[bh, k, n]
#   A = in_1 [B, H, M, K], B = in_0 [B, H, K, N], C out [B, H, M, N]
#   B=1 always; head dimension is encoded in the first grid dim.
#   Dtype cast uses C_ptr.dtype.element_ty (constexpr, resolved at JIT time).
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'K_dim'],
)
@triton.jit
def _batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K_dim,
    stride_Ah, stride_Am, stride_Ak,
    stride_Bh, stride_Bk, stride_Bn,
    stride_Ch, stride_Cm, stride_Cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)   # B=1, so this is the head index
    pid_m  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    A_head = A_ptr + pid_bh * stride_Ah
    B_head = B_ptr + pid_bh * stride_Bh
    C_head = C_ptr + pid_bh * stride_Ch

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K_dim, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_dim

        a = tl.load(
            A_head + m_offs[:, None] * stride_Am + k_offs[None, :] * stride_Ak,
            mask=(m_offs[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            B_head + k_offs[:, None] * stride_Bk + n_offs[None, :] * stride_Bn,
            mask=k_mask[:, None] & (n_offs[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)   # Triton 2.x / 3.x compatible

    # Cast float32 accumulator to the output pointer dtype (constexpr at JIT time)
    tl.store(
        C_head + m_offs[:, None] * stride_Cm + n_offs[None, :] * stride_Cn,
        acc.to(C_ptr.dtype.element_ty),
        mask=(m_offs[:, None] < M) & (n_offs[None, :] < N),
    )


# ---------------------------------------------------------------------------
# Pattern: single matmul node – simple, connected, definitely matchable
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    matmul = in_1 @ in_0
    return matmul


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_batmm(in_0, in_1):
    B, H, M, K = in_1.shape
    N = in_0.shape[-1]
    out = torch.empty((B, H, M, N), dtype=in_1.dtype, device=in_1.device)

    _batched_matmul_kernel[
        lambda meta: (B * H,
                      triton.cdiv(M, meta['BLOCK_M']),
                      triton.cdiv(N, meta['BLOCK_N']))
    ](
        in_1, in_0, out,
        M, N, K,
        in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_0.stride(1), in_0.stride(2), in_0.stride(3),
        out.stride(1),  out.stride(2),  out.stride(3),
    )

    return out


def replacement_func():
    return triton_batmm