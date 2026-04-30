import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_hardswish_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for input and weight^T
    a_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        mask_b = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    # Add bias
    mask_n = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Hardswish: x * clamp(x + 3, 0, 6) / 6
    x_plus_3 = acc + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    result = acc * relu6 / 6.0

    # Store
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, result, mask=mask_out)


@torch.fx.wrap
def fused_conv1x1_hardswish_flatten(input_tensor, weight, bias):
    B = input_tensor.shape[0]
    C_in = input_tensor.shape[1]
    C_out = weight.shape[0]

    out = torch.empty((B, C_out), dtype=input_tensor.dtype, device=input_tensor.device)

    M, K, N = B, C_in, C_out

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    matmul_hardswish_kernel[grid](
        input_tensor, weight, bias, out,
        M, N, K,
    )

    return out


def replacement_func():
    return fused_conv1x1_hardswish_flatten