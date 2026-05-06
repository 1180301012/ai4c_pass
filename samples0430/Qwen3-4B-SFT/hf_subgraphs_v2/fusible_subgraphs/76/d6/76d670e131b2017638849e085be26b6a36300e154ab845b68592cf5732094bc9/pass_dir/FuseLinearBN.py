import torch
import triton
import triton.language as tl


# ─── Match the entire model pattern: one call replaces both ops ──────────────
def pattern(in_6, in_5, in_4, in_7, in_0, in_1, in_3, in_2):
    linear = torch.nn.functional.linear(in_6, in_5, in_4)
    tmp_7  = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return linear, tmp_7


def replacement_args(in_6, in_5, in_4, in_7, in_0, in_1, in_3, in_2):
    return (in_6, in_5, in_4, in_7, in_0, in_1, in_3, in_2)


# ─── Kernel 1 : apply BN to x (inference mode) ───────────────────────────────
# scale[c] = weight[c] / sqrt(running_var[c] + eps)
# offset[c] = bias[c] - running_mean[c] * scale[c]
# out[m,c] = x[m,c]*scale[c] + offset[c]
# One kernel, sequential x/out loads; gather for scale/offset.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def _bn_inf_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid    = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask   = offsets < N
    c      = offsets // C
    mean_v = tl.load(mean_ptr  + c, mask=mask, other=0.0).to(tl.float32)
    var_v  = tl.load(var_ptr   + c, mask=mask, other=1.0).to(tl.float32)
    w_v    = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b_v    = tl.load(bias_ptr  + c, mask=mask, other=0.0).to(tl.float32)
    EPS    = 1e-05
    scale  = w_v / tl.sqrt(var_v + EPS)
    offset = b_v - mean_v * scale
    xv     = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, (xv * scale + offset).to(xv.dtype), mask=mask)


# ─── Kernel 2 : linear  (x @ w^T + b) ────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _ln_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(0, K, BLOCK_K):
        offs_k = kb + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0,
        )
        b = tl.load(
            w_ptr + offs_k[:, None] + offs_n[None, :] * K,
            mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32)

    bias_val = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = (acc + bias_val[None, :]).to(out_ptr.dtype.element_ty)

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@torch.fx.wrap
def fused_linear_bn(in_6, in_5, in_4, in_7, in_0, in_1, in_3, in_2):
    """
    Combined: F.linear(in_6, in_5, in_4) AND F.batch_norm(in_7, in_0, in_1, in_3, in_2, ...)
    in_6 : input     [M, K]
    in_5 : weight    [N, K]
    in_4 : bias      [N]
    in_7 : x         [M, C]
    in_0 : running_mean [C]
    in_1 : running_var  [C]
    in_3 : BN-weight    [C]
    in_2 : BN-bias      [C]
    """
    M  = in_6.shape[0]
    K  = in_6.shape[1]
    N  = in_5.shape[0]

    # ── output tensors ─────────────────────────────────────────────────────
    linear_out = torch.empty((M, N), dtype=in_6.dtype, device=in_6.device)
    bn_out     = torch.empty_like(in_7)

    # ── BN on in_7 ─────────────────────────────────────────────────────────
    C       = in_7.shape[1]
    N_bn    = M * C
    _bn_inf_kernel[lambda meta: (triton.cdiv(N_bn, meta['BLOCK_SIZE']),)](
        in_7, in_0, in_1, in_3, in_2, bn_out,
        C, N_bn,
    )

    # ── linear: in_6 @ in_5.T + in_4 ──────────────────────────────────────
    out = torch.empty((M, N), dtype=in_6.dtype, device=in_6.device)
    _ln_kernel[lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )](
        in_6, in_5, in_4, out,
        M, N, K,
    )

    return linear_out, bn_out


def replacement_func():
    return fused_linear_bn