import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.unfold(conv2d, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_conv1x1_unfold_kernel(
    weight_ptr,
    input_ptr,
    output_ptr,
    M,  # 128 output channels
    N,  # 1024 spatial positions (32*32)
    K,  # 256 input channels
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Tiled matmul: output[oc, hw] = sum_ic weight[oc, ic] * input[ic, hw]
    # Then write to permuted output layout
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        # Load weight tile: weight[rm, rk], weight is [M, K] contiguous
        a = tl.load(weight_ptr + rm[:, None] * K + rk[None, :],
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)

        # Load input tile: input[rk, rn], input is [K, N] contiguous
        b = tl.load(input_ptr + rk[:, None] * N + rn[None, :],
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    # Compute permuted output offsets for unfold+reshape
    # rn = h*32 + w where h,w are spatial coords in [0,31]
    h = rn // 32
    w = rn % 32
    # Unfold with kernel_size=2, stride=2:
    # k = (h%2)*2 + w%2 (position within 2x2 patch)
    # l = (h//2)*16 + w//2 (patch index in 16x16 grid)
    k_idx = (h % 2) * 2 + w % 2
    l_idx = (h // 2) * 16 + w // 2
    # Output layout: [128, 4, 256] -> offset = oc*1024 + k*256 + l
    out_offset = rm[:, None] * 1024 + k_idx[None, :] * 256 + l_idx[None, :]

    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(output_ptr + out_offset, acc.to(output_ptr.dtype.element_ty), mask=mask_out)


@torch.fx.wrap
def fused_conv1x1_unfold(weight, input_tensor):
    # weight: [128, 256, 1, 1] - 1x1 conv weight
    # input_tensor: [1, 256, 32, 32] - input activation
    # output: [1, 128, 4, 256] - fused conv + unfold + reshape

    M = 128   # output channels
    N = 1024  # 32*32 spatial positions
    K = 256   # input channels

    output = torch.empty(1, M, 4, 256, dtype=input_tensor.dtype, device=input_tensor.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    fused_conv1x1_unfold_kernel[grid](
        weight,
        input_tensor,
        output,
        M, N, K,
    )

    return output


def replacement_func():
    return fused_conv1x1_unfold