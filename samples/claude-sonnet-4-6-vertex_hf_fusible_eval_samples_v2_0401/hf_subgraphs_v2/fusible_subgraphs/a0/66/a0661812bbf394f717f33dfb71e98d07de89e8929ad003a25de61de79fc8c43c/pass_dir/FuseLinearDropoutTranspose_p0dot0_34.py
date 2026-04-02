"""
Pass: FuseLinearDropoutTranspose_p0dot0_34
Fuses: linear -> dropout(0.0, training=False) -> transpose(1,2)
Returns: (linear_out, transposed_out)  i.e. (tmp_3, tmp_4)

Dropout with training=False is a no-op (identity), so we skip it and
implement the linear projection via a custom Triton GEMM with bias addition,
then return both the raw result and a transposed view.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bias_kernel_p0dot0_34(
    A_ptr, W_ptr, b_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
        )
        w = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0,
        )
        acc += tl.dot(a, tl.trans(w))
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :].to(tl.float32)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(C_ptr.dtype.element_ty), mask=c_mask,
    )


@torch.fx.wrap
def _linear_triton_p0dot0_34(in_0, in_1, in_2):
    """
    Computes F.linear(in_2, in_1, in_0) via Triton GEMM.
    Returns: out [B, N_seq, N_out]  (single tensor, NOT a tuple)
    """
    B, N_seq, K = in_2.shape
    N_out = in_1.shape[0]

    x_2d = in_2.contiguous().reshape(B * N_seq, K)
    out_2d = torch.empty(B * N_seq, N_out, dtype=in_2.dtype, device=in_2.device)

    grid = lambda meta: (
        triton.cdiv(B * N_seq, meta['BLOCK_M']),
        triton.cdiv(N_out, meta['BLOCK_N']),
    )

    _gemm_bias_kernel_p0dot0_34[grid](
        x_2d, in_1, in_0, out_2d,
        B * N_seq, N_out, K,
        x_2d.stride(0), x_2d.stride(1),
        in_1.stride(0), in_1.stride(1),
        out_2d.stride(0), out_2d.stride(1),
    )

    return out_2d.reshape(B, N_seq, N_out)


def _replacement_p0dot0_34(in_0, in_1, in_2):
    """
    NOT wrapped – FX traces through this, producing two separate output nodes:
      node1 = _linear_triton_p0dot0_34(...)   ← wrapped leaf
      node2 = node1.transpose(1, 2)           ← FX-visible op
    Return (node1, node2) maps to pattern's (tmp_3, tmp_4).
    """
    out = _linear_triton_p0dot0_34(in_0, in_1, in_2)
    transposed = out.transpose(1, 2)
    return out, transposed


# ── Pattern ─────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return _replacement_p0dot0_34