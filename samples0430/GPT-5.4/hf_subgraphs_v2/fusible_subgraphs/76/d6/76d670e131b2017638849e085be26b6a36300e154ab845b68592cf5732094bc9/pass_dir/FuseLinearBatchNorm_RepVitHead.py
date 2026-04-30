import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    linear = torch.nn.functional.linear(in_6, in_5, in_4)
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return (linear, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_b,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    for k in range(0, k_tiles):
        k_offsets = k * BLOCK_K + offs_k
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=(offs_n[None, :] < N) & (k_offsets[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(bias_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _batch_norm_inference_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    C,
    stride_xm,
    stride_xc,
    stride_om,
    stride_oc,
    eps,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cols < C

    x = tl.load(x_ptr + pid_m * stride_xm + cols * stride_xc, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    inv_std = tl.rsqrt(var + eps)
    y = (x - mean) * inv_std
    y = y * weight + bias

    tl.store(out_ptr + pid_m * stride_om + cols * stride_oc, y, mask=mask)


@torch.fx.wrap
def _repvit_head_linear_batch_norm_dispatch(mean, var, bn_bias, bn_weight, linear_bias, linear_weight, linear_input, bn_input):
    m = linear_input.shape[0]
    k = linear_input.shape[1]
    n = linear_weight.shape[0]

    out_linear = torch.empty((m, n), device=linear_input.device, dtype=linear_input.dtype)

    linear_grid = lambda META: (
        triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),
    )
    _linear_kernel[linear_grid](
        linear_input,
        linear_weight,
        linear_bias,
        out_linear,
        m,
        n,
        k,
        linear_input.stride(0),
        linear_input.stride(1),
        linear_weight.stride(0),
        linear_weight.stride(1),
        linear_bias.stride(0),
        out_linear.stride(0),
        out_linear.stride(1),
    )

    bn_m = bn_input.shape[0]
    bn_c = bn_input.shape[1]
    out_bn = torch.empty((bn_m, bn_c), device=bn_input.device, dtype=bn_input.dtype)

    bn_grid = (bn_m, triton.cdiv(bn_c, 128))
    _batch_norm_inference_kernel[bn_grid](
        bn_input,
        mean,
        var,
        bn_weight,
        bn_bias,
        out_bn,
        bn_m,
        bn_c,
        bn_input.stride(0),
        bn_input.stride(1),
        out_bn.stride(0),
        out_bn.stride(1),
        1e-5,
        BLOCK_C=128,
    )

    return (out_linear, out_bn)


def _repvit_head_linear_batch_norm(mean, var, bn_bias, bn_weight, linear_bias, linear_weight, linear_input, bn_input):
    outs = _repvit_head_linear_batch_norm_dispatch(mean, var, bn_bias, bn_weight, linear_bias, linear_weight, linear_input, bn_input)
    return outs[0], outs[1]


def replacement_func():
    return _repvit_head_linear_batch_norm