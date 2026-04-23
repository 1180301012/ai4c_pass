import torch
import triton
import triton.language as tl


# Match exactly: conv2d -> hardswish(inplace=True) -> flatten(1, -1)
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit

def fused_conv1x1_hardswish_flatten_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_offsets = k + offs_k

        a_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offsets[None, :] * stride_xk
        b_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offsets[:, None] * stride_wk

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & (k_offsets[:, None] < K),
            other=0.0,
        )

        acc += tl.dot(a, b)
        k += BLOCK_K

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # hardswish(x) = x * clamp(x + 3, 0, 6) / 6
    tmp = acc + 3.0
    tmp = tl.maximum(0.0, tl.minimum(6.0, tmp))
    acc = acc * tmp * (1.0 / 6.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@torch.fx.wrap
def fused_conv1x1_hardswish_flatten(in_0, in_1, in_2):
    m = in_2.shape[0]
    k = in_2.shape[1]
    n = in_1.shape[0]

    out = torch.empty((m, n), device=in_2.device, dtype=in_2.dtype)

    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_M"]),
        triton.cdiv(n, META["BLOCK_N"]),
    )

    fused_conv1x1_hardswish_flatten_kernel[grid](
        in_2,
        in_1,
        in_0,
        out,
        m,
        n,
        k,
        in_2.stride(0),
        in_2.stride(1),
        in_1.stride(0),
        in_1.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


def replacement_func():
    return fused_conv1x1_hardswish_flatten