"""
Pass for GAE graph (bfloat16 + float16):
  in_0 = activation [N, K], in_1 = weight [out, K]
Matches pattern:
  tmp_2 = in_3.pow_(-0.5)
  tmp_3 = tmp_2.__eq__(inf)
  tmp_4 = tmp_2.masked_fill_(tmp_3, 0)
  tmp_5 = tmp_2[in_5];  tmp_5 = tmp_5 * in_4
  tmp_7 = tmp_2[in_2];  tmp_6 = tmp_5 * tmp_7
  linear = F.linear(in_0, in_1, None)
  return (tmp_8, linear)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Row/col scale kernel (fused inverse_sqrt + gather + multiply)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],
)
@triton.jit
def rowcol_scale_kernel(
    deg_ptr,    # bfloat16/float16 [N_nodes]
    ew_ptr,     # bfloat16/float16 [n_pairs]
    col_ptr,    # int64 [n_pairs]
    row_ptr,    # int64 [n_pairs]
    out_row_ptr,  # bfloat16/float16 [n_pairs]
    out_col_ptr,  # bfloat16/float16 [n_pairs]
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Gather deg indices and load
    deg_idx = tl.load(col_ptr + offsets, mask=mask, other=0)
    deg     = tl.load(deg_ptr + deg_idx, mask=mask, other=1.0)
    ew_val  = tl.load(ew_ptr  + offsets, mask=mask, other=1.0)
    # Handle zero-degrees and infinity as the masking branch
    inv_sqrt = tl.where(deg == 0.0, 1.0, tl.rsqrt(tl.where(deg >= 1e10, 1.0, deg)))
    # row scale
    out_row = tl.load(row_ptr + offsets, mask=mask, other=0)
    out_row = tl.where(out_row == 0, \
                      tl.where(deg >= 1e10, inv_sqrt, 0.0), ew_val * inv_sqrt)
    # col scale
    out_col = tl.load(deg_ptr + out_row, mask=mask, other=1.0)
    out_col = tl.where(out_row == 0, \
                      tl.where(deg >= 1e10, 1.0, tl.rsqrt(deg)), ew_val / tl.sqrt(deg))

    tl.store(out_row_ptr + offsets, out_row, mask=mask)
    tl.store(out_col_ptr + offsets, out_col, mask=mask)


# ---------------------------------------------------------------------------
# Linear layer matmul kernel  (x @ w.T)
# x  : [M, K]  → treated as [M, K] contiguous
# w  : [out, K]
# out: [M, out]
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
    ],
    key=['M', 'N_out', 'K'],
)
@triton.jit
def matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N_out, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        rk = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a = tl.load(x_ptr + rm[:, None] * K + rk[None, :],
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        # b has layout [N_out, K]; transpose to [K, N_out]
        b = tl.load(w_ptr + rn[None, :] * K + rk[:, None],
                    mask=(rn[None, :] < N_out) & (rk[:, None] < K), other=0.0)
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

    # Cast result back to input dtype before storing
    out = acc.to(x_ptr.dtype.element_ty)
    tl.store(out_ptr + rm[:, None] * N_out + rn[None, :],
             out,
             mask=(rm[:, None] < M) & (rn[None, :] < N_out))


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_row_col_scale_linear_gae(deg, edge_weight, col, row, x, w):
    """
    deg      : [N_nodes]              bfloat16/float16  on cuda
    edge_weight: [n_pairs]            bfloat16/float16  on cuda
    col      : [n_pairs]              int64             on cuda
    row      : [n_pairs]              int64             on cuda
    x        : [N, K]                 activation        (cpu for GAE)
    w        : [out, K]               weight            (cpu for GAE)
    """
    n_pairs = edge_weight.numel()

    out_row = torch.empty(n_pairs, dtype=edge_weight.dtype, device=edge_weight.device)
    out_col = torch.empty(n_pairs, dtype=edge_weight.dtype, device=edge_weight.device)

    grid_scale = lambda meta: (triton.cdiv(n_pairs, meta['BLOCK_SIZE']),)
    rowcol_scale_kernel[grid_scale](
        deg, edge_weight, col, row,
        out_row, out_col,
        n_pairs,
    )

    rows = row  # [n_pairs]   (this tensor IS row_ptr, renamed for clarity)

    M      = x.shape[0]
    K      = x.shape[1]
    N_out  = w.shape[0]

    x_gpu = x.to(device=edge_weight.device)
    w_gpu = w.to(device=edge_weight.device)

    out_linear = torch.empty((M, N_out), dtype=x.dtype, device=x.device)

    grid_linear = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_out, meta['BLOCK_N']),
    )
    matmul_kernel[grid_linear](
        x_gpu, w_gpu, out_linear,
        M, N_out, K,
    )

    return (out_row, out_col, out_linear)


# ---------------------------------------------------------------------------
# FX interface
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_2 = in_3.pow_(-0.5)
    tmp_3 = tmp_2.__eq__(inf)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 0)
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
    linear = torch.nn.functional.linear(in_0, in_1, None)
    return (tmp_8, linear)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # in_0 = activation [N,K], in_1 = weight [out,K]
    # in_2 = col, in_3 = deg, in_4 = edge_weight, in_5 = row
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_row_col_scale_linear_gae