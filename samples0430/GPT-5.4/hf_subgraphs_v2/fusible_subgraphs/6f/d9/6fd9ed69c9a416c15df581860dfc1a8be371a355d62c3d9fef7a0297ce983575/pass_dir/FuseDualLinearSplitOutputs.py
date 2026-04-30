import torch
import triton
import triton.language as tl


def pattern(x, w, b):
    out = torch.nn.functional.linear(x, w, b)
    return out


def replacement_args(x, w, b):
    return (x, w, b)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
    ],
    key=["M_TOTAL", "N", "K"],
)
@triton.jit
def linear_bias_kernel(
    a_ptr,
    w_ptr,
    b_ptr,
    c_ptr,
    M_TOTAL,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_b,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M_TOTAL, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m_total = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    batch_idx = offs_m_total // stride_am
    row_idx = offs_m_total % stride_am

    a_ptrs = a_ptr + batch_idx[:, None] * stride_ab + row_idx[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m_total[:, None] < M_TOTAL) & ((k0 + offs_k)[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, w)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(b_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = c_ptr + batch_idx[:, None] * stride_cb + row_idx[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m_total[:, None] < M_TOTAL) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@torch.fx.wrap
def fused_linear_bias(x, w, b):
    n = w.shape[0]
    k = x.shape[-1]
    if type(x) is not torch.Tensor:
        if x.dim() == 2:
            return torch.empty((x.shape[0], n), device=x.device, dtype=x.dtype)
        return torch.empty((x.shape[0], x.shape[1], n), device=x.device, dtype=x.dtype)

    if x.dim() == 2:
        m = x.shape[0]
        stride_ab = 0
        stride_am = x.stride(0)
        stride_ak = x.stride(1)
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        stride_cb = 0
        stride_cm = out.stride(0)
        stride_cn = out.stride(1)
        m_total = m
    else:
        batch = x.shape[0]
        m = x.shape[1]
        stride_ab = x.stride(0)
        stride_am = x.stride(1)
        stride_ak = x.stride(2)
        out = torch.empty((batch, m, n), device=x.device, dtype=x.dtype)
        stride_cb = out.stride(0)
        stride_cm = out.stride(1)
        stride_cn = out.stride(2)
        m_total = batch * m

    grid = lambda meta: (triton.cdiv(m_total, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    linear_bias_kernel[grid](
        x,
        w,
        b,
        out,
        m_total,
        n,
        k,
        stride_ab,
        stride_am,
        stride_ak,
        w.stride(0),
        w.stride(1),
        b.stride(0),
        stride_cb,
        stride_cm,
        stride_cn,
    )
    return out


def replacement_func():
    return fused_linear_bias