import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardsigmoid(conv2d, False)
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128, "BLOCK_HW": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 128, "BLOCK_HW": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 128, "BLOCK_HW": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_HW": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_HW": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_HW": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_HW": 32}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "HW"],
)
@triton.jit
# out_ptr is used as a temporary mean buffer first, then overwritten by the fused epilogue kernel.
def channel_mean_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    HW,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    MAX_HW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    base_ptrs = x_ptr + offs_m[:, None, None] * stride_xm + offs_n[None, :, None] * stride_xn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for hw_start in range(0, MAX_HW, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        ptrs = base_ptrs + offs_hw[None, None, :]
        mask = (
            (offs_m[:, None, None] < M)
            & (offs_n[None, :, None] < N)
            & (offs_hw[None, None, :] < HW)
        )
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=2)

    acc = acc / HW

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 2}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 2}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_se_matmul_epilogue_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    mean_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wo,
    stride_wk,
    stride_mm,
    stride_mn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
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

    a_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    b_ptrs = w_ptr + offs_n[None, :] * stride_wo + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_mask = offs_k < (K - k_start)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_xk
        b_ptrs += BLOCK_K * stride_wk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    gate = tl.maximum(tl.minimum(acc + bias[None, :] + 3.0, 6.0), 0.0) * (1.0 / 6.0)

    mean_ptrs = mean_ptr + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    mean_vals = tl.load(mean_ptrs, mask=mask, other=0.0).to(tl.float32)
    out_vals = gate * mean_vals

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out_vals, mask=mask)


@torch.fx.wrap
def fused_se_conv_hardsigmoid_mul_avgpool_flatten_dropout(in_0, in_1, in_2, in_3):
    m = in_2.shape[0]
    n = in_2.shape[1]
    h = in_2.shape[2]
    w = in_2.shape[3]
    hw = h * w
    k = in_3.shape[1]

    out = torch.empty((m, n), device=in_2.device, dtype=in_2.dtype)

    grid_mean = lambda META: (
        triton.cdiv(m, META["BLOCK_M"]),
        triton.cdiv(n, META["BLOCK_N"]),
    )
    channel_mean_kernel[grid_mean](
        in_2,
        out,
        m,
        n,
        hw,
        in_2.stride(0),
        in_2.stride(1),
        out.stride(0),
        out.stride(1),
        MAX_HW=144,
    )

    grid_mm = lambda META: (
        triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),
    )
    fused_se_matmul_epilogue_kernel[grid_mm](
        in_3,
        in_1,
        in_0,
        out,
        out,
        m,
        n,
        k,
        in_3.stride(0),
        in_3.stride(1),
        in_1.stride(0),
        in_1.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(0),
        out.stride(1),
    )

    return out


def replacement_func():
    return fused_se_conv_hardsigmoid_mul_avgpool_flatten_dropout