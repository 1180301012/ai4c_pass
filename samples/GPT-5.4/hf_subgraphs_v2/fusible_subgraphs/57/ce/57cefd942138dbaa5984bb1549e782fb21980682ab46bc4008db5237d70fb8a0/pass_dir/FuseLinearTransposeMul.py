import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear.transpose(-1, -2)
    tmp_4 = in_3 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
    ],
    key=["M_DIM", "N_DIM", "K_DIM"],
)
@triton.jit
def fused_linear_transpose_mul_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    u_ptr,
    out_ptr,
    B_DIM,
    M_DIM,
    N_DIM,
    K_DIM,
    stride_xb,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ub,
    stride_un,
    stride_um,
    stride_ob,
    stride_on,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M_DIM, BLOCK_M)
    num_pid_n = tl.cdiv(N_DIM, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + batch_id * stride_xb + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_DIM, BLOCK_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M_DIM) & (offs_k[None, :] < K_DIM),
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=(offs_k[:, None] < K_DIM) & (offs_n[None, :] < N_DIM),
            other=0.0,
        )
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        offs_k += BLOCK_K

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N_DIM, other=0.0)
    acc = acc + bias[None, :]

    u_ptrs = u_ptr + batch_id * stride_ub + offs_n[None, :] * stride_un + offs_m[:, None] * stride_um
    u = tl.load(
        u_ptrs,
        mask=(offs_m[:, None] < M_DIM) & (offs_n[None, :] < N_DIM),
        other=0.0,
    )
    out = acc * u

    out_ptrs = out_ptr + batch_id * stride_ob + offs_n[None, :] * stride_on + offs_m[:, None] * stride_om
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < M_DIM) & (offs_n[None, :] < N_DIM),
    )


@torch.fx.wrap
def fused_linear_transpose_mul(bias, weight, x, u):
    b_dim = x.shape[0]
    m_dim = x.shape[1]
    k_dim = x.shape[2]
    n_dim = weight.shape[0]

    out = torch.empty_like(u)

    grid = lambda META: (
        triton.cdiv(m_dim, META["BLOCK_M"]) * triton.cdiv(n_dim, META["BLOCK_N"]),
        b_dim,
    )

    fused_linear_transpose_mul_kernel[grid](
        x,
        weight,
        bias,
        u,
        out,
        b_dim,
        m_dim,
        n_dim,
        k_dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        u.stride(0),
        u.stride(1),
        u.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out


def replacement_func():
    return fused_linear_transpose_mul