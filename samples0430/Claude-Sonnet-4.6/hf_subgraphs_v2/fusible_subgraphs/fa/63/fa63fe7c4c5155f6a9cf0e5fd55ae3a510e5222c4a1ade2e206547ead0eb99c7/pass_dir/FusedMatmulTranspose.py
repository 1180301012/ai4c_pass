import torch
import triton
import triton.language as tl

# ===========================================================================
# Fused pass: matmul (in_1 @ in_0)  +  in_2[:,:,1:,:].transpose(-1,-2)
#
# Two Triton CUDA kernels from ONE Python wrapper call, eliminating one full
# Python-dispatch / GPU-idle overhead period versus having two separate ops.
#
# Dtype is handled via OUT_DTYPE constexpr so the float32 accumulator is
# correctly cast to float16 / bfloat16 / float32 at store time.
# ===========================================================================

# ---------------------------------------------------------------------------
# torch.dtype → triton dtype map  (module-level, no overhead at call time)
# ---------------------------------------------------------------------------
_DTYPE_MAP = {
    torch.float32:  tl.float32,
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
}

_ST_BLOCK = 512   # block size for the slice-transpose kernel


# ---------------------------------------------------------------------------
# Kernel A – Slice + Transpose:  out[bh, k, n] = in2[bh, n+1, k]
# (B=1, contiguous in_2: stride_in=K, stride_ik=1)
# ---------------------------------------------------------------------------
@triton.jit
def _slice_transpose_kernel(
    in_ptr,
    out_ptr,
    H,
    N,
    K,
    N_out,
    KN_out,
    BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid    = tl.program_id(1)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < KN_out

    n_idx = offs % N_out
    k_idx = offs // N_out

    in_offs = pid_bh * (N * K) + (n_idx + 1) * K + k_idx
    data = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
    tl.store(out_ptr + pid_bh * KN_out + offs, data, mask=mask)


# ---------------------------------------------------------------------------
# Kernel B – Batched GEMM:  C[bh, m, n] = sum_k A[bh, m, k] * B[bh, k, n]
# Accumulates in float32; stores in OUT_DTYPE (constexpr).
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
    OUT_DTYPE: tl.constexpr,   # tl.float32 / tl.float16 / tl.bfloat16
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)   # B=1, so just head index
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
        acc += tl.dot(a, b)   # += form compatible with Triton 2.x and 3.x

    # Explicit cast: acc is float32, output may be float16 / bfloat16
    tl.store(
        C_head + m_offs[:, None] * stride_Cm + n_offs[None, :] * stride_Cn,
        acc.to(OUT_DTYPE),
        mask=(m_offs[:, None] < M) & (n_offs[None, :] < N),
    )


# ---------------------------------------------------------------------------
# Pattern: matmul  +  in_2 slice+transpose  (2 observable outputs)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    matmul = in_1 @ in_0
    tmp_2  = in_2[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))]
    tmp_3  = tmp_2.transpose(-1, -2)
    return matmul, tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Fused wrapper: ONE Python call → TWO async Triton kernels
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_matmul_transpose(in_0, in_1, in_2):
    # ---- Batched GEMM -------------------------------------------------------
    B, H, M, K = in_1.shape
    N          = in_0.shape[-1]
    out_mm     = torch.empty((B, H, M, N), dtype=in_1.dtype, device=in_1.device)
    out_dtype  = _DTYPE_MAP[in_1.dtype]

    _batched_matmul_kernel[
        lambda meta: (B * H,
                      triton.cdiv(M, meta['BLOCK_M']),
                      triton.cdiv(N, meta['BLOCK_N']))
    ](
        in_1, in_0, out_mm,
        M, N, K,
        in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_0.stride(1), in_0.stride(2), in_0.stride(3),
        out_mm.stride(1), out_mm.stride(2), out_mm.stride(3),
        OUT_DTYPE=out_dtype,
    )

    # ---- Slice + Transpose --------------------------------------------------
    N2, K2  = in_2.shape[2], in_2.shape[3]
    N_out   = N2 - 1
    KN_out  = K2 * N_out
    out_tr  = torch.empty((B, H, K2, N_out), dtype=in_2.dtype, device=in_2.device)

    _slice_transpose_kernel[
        (B * H, (KN_out + _ST_BLOCK - 1) // _ST_BLOCK)
    ](
        in_2, out_tr,
        H, N2, K2, N_out, KN_out,
        BLOCK=_ST_BLOCK,
    )

    return out_mm, out_tr


def replacement_func():
    return fused_matmul_transpose