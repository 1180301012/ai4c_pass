import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_kernel(
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
    for k_tile in range(0, k_tiles):
        k_mask = k_tile * BLOCK_K + offs_k
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (k_mask[:, None] < K), other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(bias_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def batch_norm_inference_kernel(
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

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs_c < C

    x = tl.load(x_ptr + pid_m * stride_xm + offs_c * stride_xc, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * tl.rsqrt(var + eps)
    y = y * weight + bias

    tl.store(out_ptr + pid_m * stride_om + offs_c * stride_oc, y, mask=mask)


@torch.fx.wrap
def repvit_head_dispatch(*args):
    route = args[-1]

    if route == "linear":
        linear_bias, linear_weight, linear_input = args[0], args[1], args[2]
        m = linear_input.shape[0]
        k = linear_input.shape[1]
        n = linear_weight.shape[0]
        out = torch.empty((m, n), device=linear_input.device, dtype=linear_input.dtype)
        grid = lambda META: (
            triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),
        )
        linear_kernel[grid](
            linear_input,
            linear_weight,
            linear_bias,
            out,
            m,
            n,
            k,
            linear_input.stride(0),
            linear_input.stride(1),
            linear_weight.stride(0),
            linear_weight.stride(1),
            linear_bias.stride(0),
            out.stride(0),
            out.stride(1),
        )
        return out

    mean, var, bn_bias, bn_weight, bn_input = args[0], args[1], args[2], args[3], args[4]
    m = bn_input.shape[0]
    c = bn_input.shape[1]
    out = torch.empty((m, c), device=bn_input.device, dtype=bn_input.dtype)
    batch_norm_inference_kernel[(m, triton.cdiv(c, 128))](
        bn_input,
        mean,
        var,
        bn_weight,
        bn_bias,
        out,
        m,
        c,
        bn_input.stride(0),
        bn_input.stride(1),
        out.stride(0),
        out.stride(1),
        1e-5,
        BLOCK_C=128,
    )
    return out