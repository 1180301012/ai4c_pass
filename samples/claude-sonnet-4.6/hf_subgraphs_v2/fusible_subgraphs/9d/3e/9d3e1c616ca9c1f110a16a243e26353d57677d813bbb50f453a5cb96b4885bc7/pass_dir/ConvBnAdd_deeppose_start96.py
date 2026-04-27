"""
Pass: Fuse conv2d (1x1) + batch_norm (inference) + residual add
Targets: deeppose_resnet_101 start96_end99_0 graphs
Pattern: conv2d(in_6, in_4) -> bn(in_0,in_1,in_3,in_2) -> result += in_5
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
def _conv1x1_bn_add_kernel(
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
    Treats the spatial+batch dimension as M = N*H*W rows,
    Cin as K, Cout as N_cols.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # Decompose linear spatial index m into (batch_n, hw)
    n_idx  = offs_m // HW   # batch index
    hw_idx = offs_m - n_idx * HW  # spatial flat index (h*W + w)

    # Accumulate in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, Cin, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # x[batch, channel, hw] in NCHW layout
        x_offs = n_idx[:, None] * (Cin * HW) + offs_k[None, :] * HW + hw_idx[:, None]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < Cin)
        x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)

        # w[oc, ic] -- weight shape [Cout, Cin, 1, 1] -> stride Cin
        w_offs = offs_n[:, None] * Cin + offs_k[None, :]
        w_mask = (offs_n[:, None] < Cout) & (offs_k[None, :] < Cin)
        w = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0).to(tl.float32)

        # acc += x @ w.T  => (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N)
        acc = tl.dot(x, tl.trans(w), acc)

    # Load BN stats (per output channel)
    bn_mean  = tl.load(bn_mean_ptr  + offs_n, mask=offs_n < Cout).to(tl.float32)
    bn_var   = tl.load(bn_var_ptr   + offs_n, mask=offs_n < Cout).to(tl.float32)
    bn_scale = tl.load(bn_scale_ptr + offs_n, mask=offs_n < Cout).to(tl.float32)
    bn_bias_v = tl.load(bn_bias_ptr + offs_n, mask=offs_n < Cout).to(tl.float32)

    # BN: scale = gamma/sqrt(var+eps), shift = beta - mean*scale
    scale = bn_scale * tl.rsqrt(bn_var + eps)
    shift = bn_bias_v - bn_mean * scale

    acc = acc * scale[None, :] + shift[None, :]

    # Load and add residual: res[batch, oc, hw] in NCHW layout
    res_offs = n_idx[:, None] * (Cout * HW) + offs_n[None, :] * HW + hw_idx[:, None]
    res_mask = (offs_m[:, None] < M) & (offs_n[None, :] < Cout)
    res = tl.load(res_ptr + res_offs, mask=res_mask, other=0.0).to(tl.float32)
    acc = acc + res

    # Store output in same dtype as residual
    tl.store(out_ptr + res_offs, acc, mask=res_mask)


@torch.fx.wrap
def conv1x1_bn_add_v1(conv_weight, x, bn_mean, bn_var, bn_scale, bn_bias, residual):
    """
    Wrapper for Pattern A:
      args = (conv_weight=in_4, x=in_6, bn_mean=in_0, bn_var=in_1,
              bn_scale=in_3, bn_bias=in_2, residual=in_5)
    """
    N   = x.shape[0]
    Cin = x.shape[1]
    H   = x.shape[2]
    W   = x.shape[3]
    Cout = conv_weight.shape[0]
    HW   = H * W
    M    = N * HW

    out = torch.empty_like(residual)

    grid = lambda META: (
        triton.cdiv(M,    META['BLOCK_M']),
        triton.cdiv(Cout, META['BLOCK_N']),
    )

    _conv1x1_bn_add_kernel[grid](
        x, conv_weight,
        bn_mean, bn_var, bn_scale, bn_bias,
        residual, out,
        M, Cin, Cout, HW,
        1e-5,
    )
    return out


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 += in_5
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # (conv_weight, x, bn_mean, bn_var, bn_scale, bn_bias, residual)
    return (in_4, in_6, in_0, in_1, in_3, in_2, in_5)


def replacement_func():
    return conv1x1_bn_add_v1