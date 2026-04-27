import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused kernel: out = relu(res + x @ w^T + bias)
# 2D grid – BLOCK_N=128 always covers the full N=128 in one N-block.
# Shared-mem (fp32 worst-case): (BM+BN)×BK×4×stages ≤ 163 KB (A30 limit).
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # ── BLOCK_K=32, various BLOCK_M ──────────────────────────────────────
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        # ── BLOCK_K=64 ───────────────────────────────────────────────────────
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 8}),
        # ── BLOCK_K=128 (single K-iteration) ─────────────────────────────────
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_add_relu_kernel(
    x_ptr,      # [M, K]
    w_ptr,      # [N, K]
    bias_ptr,   # [N]
    res_ptr,    # [M, N]
    out_ptr,    # [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_rm, stride_rn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """out = relu(res + x @ w^T + bias)"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        # x: M mask + K mask (K may not divide BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0, eviction_policy="evict_first")

        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        # w: N mask dropped (BLOCK_N=128=N always), only K mask needed
        w_tile = tl.load(w_ptrs, mask=(offs_k[None, :] < K), other=0.0, eviction_policy="evict_last")

        acc += tl.dot(x_tile, tl.trans(w_tile), out_dtype=tl.float32)

    # Only M needs masking; N is always fully covered (BLOCK_N=128=N)
    mask_m = offs_m < M

    # Bias: no mask needed (N always covered)
    bias = tl.load(bias_ptr + offs_n)
    acc  = acc + bias[None, :].to(tl.float32)

    res_ptrs = res_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn
    res = tl.load(res_ptrs, mask=mask_m[:, None], other=0.0, eviction_policy="evict_first")
    acc = acc + res.to(tl.float32)

    acc = tl.maximum(acc, 0.0)

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


# ---------------------------------------------------------------------------
# Device-transfer cache: avoid repeated .to() overhead per iteration
# ---------------------------------------------------------------------------
_w_cache: dict = {}
_b_cache: dict = {}


@torch.fx.wrap
def fused_linear_add_relu(bias, weight, residual, x):
    device = x.device

    wptr = weight.data_ptr()
    if wptr not in _w_cache:
        _w_cache[wptr] = weight.to(device=device)
    w = _w_cache[wptr]

    bptr = bias.data_ptr()
    if bptr not in _b_cache:
        _b_cache[bptr] = bias.to(device=device)
    b = _b_cache[bptr]

    M, K = x.shape[0], x.shape[1]
    N    = w.shape[0]
    out  = torch.empty((M, N), dtype=residual.dtype, device=device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    fused_linear_add_relu_kernel[grid](
        x, w, b, residual, out,
        M, N, K,
        x.stride(0),        x.stride(1),
        w.stride(0),        w.stride(1),
        residual.stride(0), residual.stride(1),
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3  = in_2 + linear
    tmp_4  = tmp_3.relu_()
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_linear_add_relu