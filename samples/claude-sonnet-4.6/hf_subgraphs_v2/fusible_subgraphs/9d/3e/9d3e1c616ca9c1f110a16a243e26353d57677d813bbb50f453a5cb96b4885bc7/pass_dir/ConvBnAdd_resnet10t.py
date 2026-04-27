"""
Pass: Fuse conv2d (1x1) + batch_norm (inference) + residual add
Targets: resnet10t.c3_in1k start23_end26_7 graphs
Pattern: conv2d(in_5, in_0) -> bn(in_1,in_2,in_4,in_3) -> in_6 += result
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'Cout', 'Cin'],
)
@triton.jit
def _conv1x1_bn_add_kernel_v3(
    x_ptr, w_ptr,
    bn_mean_ptr, bn_var_ptr, bn_scale_ptr, bn_bias_ptr,
    res_ptr, out_ptr,
    M, Cin, Cout, HW,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused 1x1-conv + BN (inference) + residual add.
    Input layout: NCHW.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # Decompose linear index m into (batch_n, hw)
    n_idx  = offs_m // HW
    hw_idx = offs_m - n_idx * HW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, Cin, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # x[batch, channel, hw] in NCHW
        x_offs = n_idx[:, None] * (Cin * HW) + offs_k[None, :] * HW + hw_idx[:, None]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < Cin)
        x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)

        # w[oc, ic] -- weight shape [Cout, Cin, 1, 1] -> stride Cin
        w_offs = offs_n[:, None] * Cin + offs_k[None, :]
        w_mask = (offs_n[:, None] < Cout) & (offs_k[None, :] < Cin)
        w = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0).to(tl.float32)

        acc = tl.dot(x, tl.trans(w), acc)

    # BN parameters (per output channel)
    bn_mean   = tl.load(bn_mean_ptr  + offs_n, mask=offs_n < Cout).to(tl.float32)
    bn_var    = tl.load(bn_var_ptr   + offs_n, mask=offs_n < Cout).to(tl.float32)
    bn_scale  = tl.load(bn_scale_ptr + offs_n, mask=offs_n < Cout).to(tl.float32)
    bn_bias_v = tl.load(bn_bias_ptr  + offs_n, mask=offs_n < Cout).to(tl.float32)

    scale = bn_scale * tl.rsqrt(bn_var + eps)
    shift = bn_bias_v - bn_mean * scale

    acc = acc * scale[None, :] + shift[None, :]

    # Residual add
    res_offs = n_idx[:, None] * (Cout * HW) + offs_n[None, :] * HW + hw_idx[:, None]
    res_mask = (offs_m[:, None] < M) & (offs_n[None, :] < Cout)
    res = tl.load(res_ptr + res_offs, mask=res_mask, other=0.0).to(tl.float32)
    acc = acc + res

    tl.store(out_ptr + res_offs, acc, mask=res_mask)


@torch.fx.wrap
def conv1x1_bn_add_v3(conv_weight, x, bn_mean, bn_var, bn_scale, bn_bias, residual):
    """
    Wrapper for Pattern C:
      args = (conv_weight=in_0, x=in_5, bn_mean=in_1, bn_var=in_2,
              bn_scale=in_4, bn_bias=in_3, residual=in_6)
    """
    N    = x.shape[0]
    Cin  = x.shape[1]
    H    = x.shape[2]
    W    = x.shape[3]
    Cout = conv_weight.shape[0]
    HW   = H * W
    M    = N * HW

    out = torch.empty_like(residual)

    grid = lambda META: (
        triton.cdiv(M,    META['BLOCK_M']),
        triton.cdiv(Cout, META['BLOCK_N']),
    )

    _conv1x1_bn_add_kernel_v3[grid](
        x, conv_weight,
        bn_mean, bn_var, bn_scale, bn_bias,
        residual, out,
        M, Cin, Cout, HW,
        1e-5,
    )
    return out


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_5, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    in_6 += tmp_6
    return (in_6,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # (conv_weight, x, bn_mean, bn_var, bn_scale, bn_bias, residual)
    return (in_0, in_5, in_1, in_2, in_4, in_3, in_6)


def replacement_func():
    return conv1x1_bn_add_v3