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
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
    ],
    key=["M", "N_OUT", "K", "H", "W"],
)
@triton.jit
def fused_se_conv_hsigmoid_mul_pool_flatten_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    se_ptr,
    out_ptr,
    M,
    N_OUT,
    K,
    sb0,
    sw0,
    sw1,
    sx0,
    sx1,
    sx2,
    sx3,
    ss0,
    ss1,
    so0,
    so1,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N_OUT, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)

    pooled = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in range(H):
        for w in range(W):
            x_ptrs = (
                x_ptr
                + offs_m[:, None] * sx0
                + offs_n[None, :] * sx1
                + h * sx2
                + w * sx3
            )
            x = tl.load(x_ptrs, mask=mask_mn, other=0.0)
            pooled += x.to(tl.float32)
    pooled *= 1.0 / (H * W)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = se_ptr + offs_m[:, None] * ss0 + offs_k[None, :] * ss1
        b_ptrs = weight_ptr + offs_n[:, None] * sw0 + offs_k[None, :] * sw1
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[:, None] < N_OUT) & (offs_k[None, :] < K), other=0.0)
        acc += tl.dot(a, tl.trans(b))

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N_OUT, other=0.0).to(tl.float32)
    gate = acc + bias[None, :]
    gate = gate * (1.0 / 6.0) + 0.5
    gate = tl.maximum(0.0, tl.minimum(1.0, gate))

    out = pooled * gate
    out_ptrs = out_ptr + offs_m[:, None] * so0 + offs_n[None, :] * so1
    tl.store(out_ptrs, out, mask=mask_mn)


@torch.fx.wrap
def fused_se_conv_hsigmoid_mul_pool_flatten(bias, weight, x, se):
    m = x.shape[0]
    n_out = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    k = se.shape[1]

    out = torch.empty((m, n_out), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n_out, META["BLOCK_N"]),
    )

    fused_se_conv_hsigmoid_mul_pool_flatten_kernel[grid](
        bias,
        weight,
        x,
        se,
        out,
        m,
        n_out,
        k,
        bias.stride(0),
        weight.stride(0),
        weight.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        se.stride(0),
        se.stride(1),
        out.stride(0),
        out.stride(1),
        H=h,
        W=w,
    )
    return out


def replacement_func():
    return fused_se_conv_hsigmoid_mul_pool_flatten