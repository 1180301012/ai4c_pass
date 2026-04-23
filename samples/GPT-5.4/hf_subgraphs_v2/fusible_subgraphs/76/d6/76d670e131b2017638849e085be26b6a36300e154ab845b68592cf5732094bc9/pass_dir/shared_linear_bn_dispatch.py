import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_bias_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
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

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k = 0
    while k < K:
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_C": 256}, num_stages=2, num_warps=4),
    ],
    key=["M", "C"],
)
@triton.jit
def _batch_norm_inference_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    M,
    C,
    stride_xm,
    stride_xc,
    stride_ym,
    stride_yc,
    eps,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (pid_m < M) & (offs_c < C)

    x_ptrs = x_ptr + pid_m * stride_xm + offs_c * stride_xc
    y_ptrs = y_ptr + pid_m * stride_ym + offs_c * stride_yc

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + offs_c, mask=offs_c < C, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + offs_c, mask=offs_c < C, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs_c, mask=offs_c < C, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs_c, mask=offs_c < C, other=0.0).to(tl.float32)

    y = (x - mean) * tl.rsqrt(var + eps) * weight + bias
    tl.store(y_ptrs, y, mask=mask)


@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]

    if route == "linear":
        b, w, x, _route = args
        M = x.shape[0]
        K = x.shape[1]
        N = w.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
        _linear_bias_kernel[grid](
            x,
            w,
            b,
            out,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            out.stride(0),
            out.stride(1),
        )
        return out

    if route == "bn":
        mean, var, bias, weight, x, _route = args
        M = x.shape[0]
        C = x.shape[1]
        out = torch.empty_like(x)
        grid = lambda META: (M, triton.cdiv(C, META["BLOCK_C"]))
        _batch_norm_inference_kernel[grid](
            x,
            mean,
            var,
            weight,
            bias,
            out,
            M,
            C,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            1e-05,
        )
        return out

    raise RuntimeError("unknown route")


def replacement_func():
    return shared_dispatch