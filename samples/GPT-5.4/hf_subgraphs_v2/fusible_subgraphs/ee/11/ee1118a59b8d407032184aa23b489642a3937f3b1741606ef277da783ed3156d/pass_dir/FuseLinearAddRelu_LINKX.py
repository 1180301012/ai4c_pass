import torch
import triton
import triton.language as tl


# Pattern matching function
# Must mirror the source graph exactly.
def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return tmp_4


# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_linear_add_relu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    r_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_rm,
    stride_rn,
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
    # linear uses weight with shape [N, K], equivalent to x @ w^T
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_iter = 0
    while k_iter < K:
        x_mask = (offs_m[:, None] < M) & ((k_iter + offs_k)[None, :] < K)
        w_mask = ((k_iter + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        k_iter += BLOCK_K

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    r_ptrs = r_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn
    r_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    r = tl.load(r_ptrs, mask=r_mask, other=0.0).to(tl.float32)
    acc = acc + r
    acc = tl.maximum(acc, 0.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def fused_linear_add_relu(in_0, in_1, in_2, in_3):
    # in_0: bias [N]
    # in_1: weight [N, K]
    # in_2: residual [M, N]
    # in_3: input [M, K]
    M = in_3.shape[0]
    K = in_3.shape[1]
    N = in_1.shape[0]

    out = torch.empty((M, N), device=in_3.device, dtype=in_3.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _fused_linear_add_relu_kernel[grid](
        in_3,
        in_1,
        in_0,
        in_2,
        out,
        M,
        N,
        K,
        in_3.stride(0),
        in_3.stride(1),
        in_1.stride(0),
        in_1.stride(1),
        in_2.stride(0),
        in_2.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_linear_add_relu