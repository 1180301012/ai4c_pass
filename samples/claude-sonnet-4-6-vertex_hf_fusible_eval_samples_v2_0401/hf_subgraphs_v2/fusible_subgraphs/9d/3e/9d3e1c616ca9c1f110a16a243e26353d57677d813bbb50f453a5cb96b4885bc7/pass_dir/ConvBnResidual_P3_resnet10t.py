"""
Pass 3: Conv2D (1x1) + BatchNorm (inference) + Residual Add — resnet10t variant
  in_0: conv_weight, in_1: running_mean, in_2: running_var, in_3: bn_bias, in_4: bn_weight
  in_5: conv_input, in_6: residual
Strategy: single Triton kernel: GEMM (NCHW) + BN + residual add.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N_out', 'K'],
)
@triton.jit
def gemm_bn_res_kernel_p3(
    x_ptr, w_ptr, res_ptr,
    mean_ptr, var_ptr, bn_w_ptr, bn_b_ptr,
    out_ptr,
    M, N_out, K, HW,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)
    m_offs = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = n_pid * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < N_out
    n_batch = m_offs // HW
    hw      = m_offs % HW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        x_offs = n_batch[None, :] * (K * HW) + k_offs[:, None] * HW + hw[None, :]
        x_KM = tl.load(x_ptr + x_offs, mask=k_mask[:, None] & m_mask[None, :], other=0.0).to(tl.float32)
        w_offs = n_offs[:, None] * K + k_offs[None, :]
        w_NK = tl.load(w_ptr + w_offs, mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.dot(tl.trans(x_KM), tl.trans(w_NK))
    mean = tl.load(mean_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + n_offs, mask=n_mask, other=1.0).to(tl.float32)
    bn_w = tl.load(bn_w_ptr + n_offs, mask=n_mask, other=1.0).to(tl.float32)
    bn_b = tl.load(bn_b_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    bn_result = bn_w[None, :] * (acc - mean[None, :]) * tl.rsqrt(var[None, :] + 1e-5) + bn_b[None, :]
    res_offs = n_batch[:, None] * (N_out * HW) + n_offs[None, :] * HW + hw[:, None]
    res_mask = m_mask[:, None] & n_mask[None, :]
    res = tl.load(res_ptr + res_offs, mask=res_mask, other=0.0)
    tl.store(out_ptr + res_offs, bn_result.to(res.dtype) + res, mask=res_mask)


@torch.fx.wrap
def conv_bn_res_p3(conv_weight, running_mean, running_var, bn_bias, bn_weight,
                   x, residual):
    N, K, H, W = x.shape
    N_out = conv_weight.shape[0]
    HW = H * W
    M  = N * HW
    w_2d = conv_weight.view(N_out, K)
    out  = residual.new_empty(residual.shape)
    grid = lambda meta: (
        (M     + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
        (N_out + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
    )
    gemm_bn_res_kernel_p3[grid](
        x, w_2d, residual,
        running_mean, running_var, bn_weight, bn_bias,
        out, M, N_out, K, HW,
    )
    return out


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_5, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_1, in_2, in_4, in_3,
                                            False, 0.1, 1e-05)
    out = in_6 + tmp_6
    return out


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


def replacement_func():
    return conv_bn_res_p3