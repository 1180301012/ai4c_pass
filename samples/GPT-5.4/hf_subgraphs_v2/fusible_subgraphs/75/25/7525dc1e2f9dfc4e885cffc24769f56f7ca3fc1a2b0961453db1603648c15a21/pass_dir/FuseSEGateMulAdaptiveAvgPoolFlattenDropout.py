import torch
import triton
import triton.language as tl


def pattern(in_2, tmp_3):
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(in_2, tmp_3):
    return (in_2, tmp_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128, "BLOCK_HW": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 128, "BLOCK_HW": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 128, "BLOCK_HW": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_HW": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_HW": 32, "GROUP_M": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_HW": 32, "GROUP_M": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_HW": 32, "GROUP_M": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_HW": 32, "GROUP_M": 4}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "HW"],
)
@triton.jit
def fused_gate_mul_mean_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    M,
    N,
    W,
    HW,
    stride_xm,
    stride_xn,
    stride_xh,
    stride_xw,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    MAX_HW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for hw_start in range(0, MAX_HW, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        h_idx = offs_hw // W
        w_idx = offs_hw - h_idx * W
        ptrs = (
            x_ptr
            + offs_m[:, None, None] * stride_xm
            + offs_n[None, :, None] * stride_xn
            + h_idx[None, None, :] * stride_xh
            + w_idx[None, None, :] * stride_xw
        )
        mask = (
            (offs_m[:, None, None] < M)
            & (offs_n[None, :, None] < N)
            & (offs_hw[None, None, :] < HW)
        )
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=2)

    gate_ptrs = gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    gate = tl.load(gate_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    out = (acc / HW) * gate

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=out_mask)


@torch.fx.wrap
def fused_se_gate_mul_adaptive_avg_pool_flatten_dropout(in_2, tmp_3):
    m = in_2.shape[0]
    n = in_2.shape[1]
    w = in_2.shape[3]
    hw = in_2.shape[2] * in_2.shape[3]

    out = torch.empty((m, n), device=in_2.device, dtype=in_2.dtype)

    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),
    )
    fused_gate_mul_mean_kernel[grid](
        in_2,
        tmp_3,
        out,
        m,
        n,
        w,
        hw,
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        tmp_3.stride(0),
        tmp_3.stride(1),
        out.stride(0),
        out.stride(1),
        MAX_HW=144,
    )
    return out


def replacement_func():
    return fused_se_gate_mul_adaptive_avg_pool_flatten_dropout