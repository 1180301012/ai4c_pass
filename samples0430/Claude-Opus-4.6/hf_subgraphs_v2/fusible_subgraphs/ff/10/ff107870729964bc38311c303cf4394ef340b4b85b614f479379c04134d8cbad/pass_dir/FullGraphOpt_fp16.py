import torch
import triton
import triton.language as tl


def pattern(x, w, b):
    conv = torch.conv2d(x, w, b, [1, 1], [0, 0], [1, 1], 1)
    return conv


def replacement_args(x, w, b):
    return (x, w, b)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(bias_ptr + rm, mask=rm < M, other=0.0)
    acc += bias[:, None]

    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc, mask=c_mask)


@torch.fx.wrap
def conv1x1_replacement(input_tensor, weight, bias):
    # input_tensor: [1, C_in, H, W], weight: [C_out, C_in, 1, 1], bias: [C_out]
    C_out = weight.shape[0]
    C_in = weight.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    N_spatial = H * W

    # Output shape: [1, C_out, H, W]
    output = torch.empty((1, C_out, H, W), device=input_tensor.device, dtype=input_tensor.dtype)

    # Matmul: weight[C_out, C_in] @ input[C_in, H*W] + bias[C_out]
    M, K, N = C_out, C_in, N_spatial
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    matmul_bias_kernel[grid](
        weight, input_tensor, bias, output,
        M, N, K,
        C_in, 1,        # stride_am, stride_ak (weight is [C_out, C_in, 1, 1] contiguous)
        N_spatial, 1,    # stride_bk, stride_bn (input viewed as [C_in, H*W])
        N_spatial, 1,    # stride_cm, stride_cn (output viewed as [C_out, H*W])
    )

    return output


def replacement_func():
    return conv1x1_replacement