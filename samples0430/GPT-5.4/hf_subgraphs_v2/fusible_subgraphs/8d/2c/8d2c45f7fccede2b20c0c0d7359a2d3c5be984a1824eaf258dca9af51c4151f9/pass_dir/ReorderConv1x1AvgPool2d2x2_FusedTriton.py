import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(conv2d, 2, 2, 0, False, True, None)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _conv1x1_avgpool2x2_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    HO,
    WO,
    H,
    W,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wn,
    stride_wk,
    stride_on,
    stride_oc,
    stride_oh,
    stride_ow,
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
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    wo_idx = offs_m % WO
    tmp = offs_m // WO
    ho_idx = tmp % HO
    batch_idx = tmp // HO

    h0 = ho_idx * 2
    w0 = wo_idx * 2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k

        x_mask_base = (offs_m[:, None] < M) & (k_idx[None, :] < K)

        x00 = tl.load(
            x_ptr
            + batch_idx[:, None] * stride_xn
            + k_idx[None, :] * stride_xc
            + h0[:, None] * stride_xh
            + w0[:, None] * stride_xw,
            mask=x_mask_base & (h0[:, None] < H) & (w0[:, None] < W),
            other=0.0,
        )
        x01 = tl.load(
            x_ptr
            + batch_idx[:, None] * stride_xn
            + k_idx[None, :] * stride_xc
            + h0[:, None] * stride_xh
            + (w0[:, None] + 1) * stride_xw,
            mask=x_mask_base & (h0[:, None] < H) & ((w0[:, None] + 1) < W),
            other=0.0,
        )
        x10 = tl.load(
            x_ptr
            + batch_idx[:, None] * stride_xn
            + k_idx[None, :] * stride_xc
            + (h0[:, None] + 1) * stride_xh
            + w0[:, None] * stride_xw,
            mask=x_mask_base & ((h0[:, None] + 1) < H) & (w0[:, None] < W),
            other=0.0,
        )
        x11 = tl.load(
            x_ptr
            + batch_idx[:, None] * stride_xn
            + k_idx[None, :] * stride_xc
            + (h0[:, None] + 1) * stride_xh
            + (w0[:, None] + 1) * stride_xw,
            mask=x_mask_base & ((h0[:, None] + 1) < H) & ((w0[:, None] + 1) < W),
            other=0.0,
        )

        pooled = (x00 + x01 + x10 + x11) * 0.25

        w_block = tl.load(
            w_ptr + k_idx[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(k_idx[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(pooled, w_block)

    out_ptrs = (
        out_ptr
        + batch_idx[:, None] * stride_on
        + offs_n[None, :] * stride_oc
        + ho_idx[:, None] * stride_oh
        + wo_idx[:, None] * stride_ow
    )
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def conv1x1_avgpool2x2_fused(weight, x):
    batch = x.shape[0]
    cin = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    cout = weight.shape[0]

    ho = (h + 1) // 2
    wo = (w + 1) // 2
    m = batch * ho * wo
    n = cout
    k = cin

    out = torch.empty((batch, cout, ho, wo), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)

    _conv1x1_avgpool2x2_kernel[grid](
        x,
        weight,
        out,
        m,
        n,
        k,
        ho,
        wo,
        h,
        w,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out


def replacement_func():
    return conv1x1_avgpool2x2_fused