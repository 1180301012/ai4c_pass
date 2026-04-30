import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'TILE_M': 16, 'TILE_N': 256, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'TILE_M': 32, 'TILE_N': 256, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'TILE_M': 16, 'TILE_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'TILE_M': 32, 'TILE_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'TILE_M': 64, 'TILE_N': 256, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'TILE_M': 16, 'TILE_N': 256, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'TILE_M': 32, 'TILE_N': 256, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
    ],
    key=['B', 'M', 'K', 'N'],
)
@triton.jit
def _fused_scaled_attn_permute_kernel(
    A_ptr, C_ptr, Out_ptr,
    B, M, K, N,
    stride_Ab, stride_Am, stride_Ak,
    stride_Cb, stride_Ck, stride_Cn,
    stride_Ob, stride_On, stride_Om,
    scale,
    IS_FP16: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel: out[b, n, m] = sum_k( softmax(0.0625 * A[b, m, :]) * C[b, :, n] )
    A:   [B, M, K]  - attention logits
    C:   [B, K, N]  - value matrix
    Out: [B, N, M]  - permuted result (matmul followed by permute)
    """
    pid_m = tl.program_id(0)   # tile over M (8192)
    pid_b = tl.program_id(1)   # batch

    m_start = pid_m * TILE_M
    m_offs  = m_start + tl.arange(0, TILE_M)
    n_offs  = tl.arange(0, TILE_N)
    m_mask  = m_offs < M
    n_mask  = n_offs < N

    # Accumulate in fp32 for numerical stability
    acc = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)

    A_base = A_ptr + pid_b * stride_Ab
    C_base = C_ptr + pid_b * stride_Cb

    # K is always small (<= 19) so BLOCK_K=32 handles it with masking in one pass
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # --- Load A tile [TILE_M, BLOCK_K] ---
        a_ptrs  = A_base + m_offs[:, None] * stride_Am + k_offs[None, :] * stride_Ak
        a_mask  = m_mask[:, None] & k_mask[None, :]
        a_vals  = tl.load(a_ptrs, mask=a_mask, other=0.0)  # fp16/bf16

        # --- Scale + row-wise softmax ---
        scaled     = a_vals * scale  # scale = 0.0625
        scaled_fp32 = scaled.to(tl.float32)
        # subtract row max for numerical stability
        row_max    = tl.max(scaled_fp32, axis=1)
        scaled_fp32 = scaled_fp32 - row_max[:, None]
        exp_vals   = tl.exp(scaled_fp32)
        row_sum    = tl.sum(exp_vals, axis=1)
        attn       = (exp_vals / row_sum[:, None]).to(a_vals.dtype)  # back to fp16/bf16

        # --- Load C tile [BLOCK_K, TILE_N] (avoids loading C redundantly per row) ---
        c_ptrs  = C_base + k_offs[:, None] * stride_Ck + n_offs[None, :] * stride_Cn
        c_mask  = k_mask[:, None] & n_mask[None, :]
        v       = tl.load(c_ptrs, mask=c_mask, other=0.0)  # [BLOCK_K, TILE_N]

        # attn: [TILE_M, BLOCK_K]  x  v.T: [TILE_N, BLOCK_K]  -> [TILE_M, TILE_N]
        if IS_FP16:
            acc += tl.dot(attn.to(tl.float16), tl.trans(v).to(tl.float16)).to(tl.float32)
        else:
            acc += tl.dot(attn.to(tl.bfloat16), tl.trans(v).to(tl.bfloat16)).to(tl.float32)

    # --- Store: Out[b, n, m] at stride_Ob + n*stride_On + m*stride_Om ---
    Out_base = Out_ptr + pid_b * stride_Ob
    out_ptrs = Out_base + n_offs[None, :] * stride_On + m_offs[:, None] * stride_Om
    out_mask = m_mask[:, None] & n_mask[None, :]

    if IS_FP16:
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)
    else:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@torch.fx.wrap
def fused_scaled_attn_permute(in_0, in_1):
    """
    Fused: scale(0.0625) + softmax + matmul + permute(0,2,1)
    in_0 : [B, M, K]  (M=8192, K=19)
    in_1 : [B, K, N]  (N=256)
    out  : [B, N, M]
    """
    B, M, K = in_0.shape
    N       = in_1.shape[2]

    out = torch.empty((B, N, M), dtype=in_0.dtype, device=in_0.device)

    is_fp16 = in_0.dtype == torch.float16

    grid = lambda meta: (triton.cdiv(M, meta['TILE_M']), B)

    _fused_scaled_attn_permute_kernel[grid](
        in_0, in_1, out,
        B, M, K, N,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
        0.0625,
        IS_FP16=is_fp16,
    )

    return out


def replacement_func():
    return fused_scaled_attn_permute