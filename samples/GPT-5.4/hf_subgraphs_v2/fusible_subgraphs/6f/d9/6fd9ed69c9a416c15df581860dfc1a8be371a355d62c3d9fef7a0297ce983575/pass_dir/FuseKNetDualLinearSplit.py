import torch
import triton
import triton.language as tl


def pattern(bias, weight, x):
    linear = torch.nn.functional.linear(x, weight, bias)
    return linear


def replacement_args(bias, weight, x):
    return (x, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=3),
    ],
    key=["M_TOTAL", "N", "K"],
)
@triton.jit
def _linear_bias_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    M_TOTAL,
    N,
    K,
    X_ROWS,
    X_STRIDE_0,
    X_STRIDE_1,
    X_STRIDE_2,
    W_STRIDE_0,
    W_STRIDE_1,
    OUT_STRIDE_0,
    OUT_STRIDE_1,
    OUT_STRIDE_2,
    HAS_BATCH: tl.constexpr,
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

    if HAS_BATCH:
        batch_idx = offs_m_total // X_ROWS
        row_idx = offs_m_total % X_ROWS
        x_base_ptr = x_ptr + batch_idx[:, None] * X_STRIDE_0 + row_idx[:, None] * X_STRIDE_1
        out_base_ptr = out_ptr + batch_idx[:, None] * OUT_STRIDE_0 + row_idx[:, None] * OUT_STRIDE_1
    else:
        x_base_ptr = x_ptr + offs_m_total[:, None] * X_STRIDE_1
        out_base_ptr = out_ptr + offs_m_total[:, None] * OUT_STRIDE_1

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        x = tl.load(
            x_base_ptr + k_idx[None, :] * X_STRIDE_2,
            mask=(offs_m_total[:, None] < M_TOTAL) & (k_idx[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :] * W_STRIDE_0 + k_idx[:, None] * W_STRIDE_1,
            mask=(offs_n[None, :] < N) & (k_idx[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(x, w)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    tl.store(
        out_base_ptr + offs_n[None, :] * OUT_STRIDE_2,
        acc,
        mask=(offs_m_total[:, None] < M_TOTAL) & (offs_n[None, :] < N),
    )


@torch.fx.wrap
def fused_linear(x, weight, bias):
    x_shape = x.shape
    x_dim = x.dim()
    k = x_shape[-1]
    n = weight.shape[0]

    if x_dim == 2:
        m_total = x_shape[0]
        x_rows = 0
        out = torch.empty((m_total, n), device=x.device, dtype=x.dtype)
        x_stride_0 = 0
        x_stride_1 = x.stride(0)
        x_stride_2 = x.stride(1)
        out_stride_0 = 0
        out_stride_1 = out.stride(0)
        out_stride_2 = out.stride(1)
        has_batch = False
    else:
        batch = x_shape[0]
        rows = x_shape[1]
        m_total = batch * rows
        x_rows = rows
        out = torch.empty((batch, rows, n), device=x.device, dtype=x.dtype)
        x_stride_0 = x.stride(0)
        x_stride_1 = x.stride(1)
        x_stride_2 = x.stride(2)
        out_stride_0 = out.stride(0)
        out_stride_1 = out.stride(1)
        out_stride_2 = out.stride(2)
        has_batch = True

    grid = lambda META: (triton.cdiv(m_total, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)

    _linear_bias_kernel[grid](
        x,
        weight,
        bias,
        out,
        m_total,
        n,
        k,
        x_rows,
        x_stride_0,
        x_stride_1,
        x_stride_2,
        weight.stride(0),
        weight.stride(1),
        out_stride_0,
        out_stride_1,
        out_stride_2,
        HAS_BATCH=has_batch,
    )

    return out


def replacement_func():
    return fused_linear