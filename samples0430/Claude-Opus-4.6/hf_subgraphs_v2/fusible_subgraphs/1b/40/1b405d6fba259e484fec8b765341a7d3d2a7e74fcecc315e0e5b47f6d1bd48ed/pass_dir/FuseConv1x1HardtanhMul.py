import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ==================== Single-batch kernel (no division/modulo) ====================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_conv1x1_single_batch_kernel(
    input_ptr, weight_ptr, bias_ptr, act_ptr, output_ptr,
    M, K, N, HW,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Single batch: input addr(m, k) = k * HW + m
    input_addrs = input_ptr + offs_k[None, :] * HW + offs_m[:, None]
    mask_input = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    input_tile = tl.load(input_addrs, mask=mask_input, other=0.0)

    # Weight tile [BLOCK_K, BLOCK_N]: weight[n, k] at addr n * K + k
    weight_addrs = weight_ptr + offs_n[None, :] * K + offs_k[:, None]
    mask_weight = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    weight_tile = tl.load(weight_addrs, mask=mask_weight, other=0.0)

    # Matmul
    acc = tl.dot(input_tile, weight_tile)

    # Bias
    mask_n = offs_n < N
    bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias_vals[None, :].to(tl.float32)

    # Activation: addr(m, n) = n * HW + m
    act_addrs = act_ptr + offs_n[None, :] * HW + offs_m[:, None]
    mask_out = (offs_m[:, None] < M) & mask_n[None, :]
    act_vals = tl.load(act_addrs, mask=mask_out, other=0.0).to(tl.float32)

    # Hardtanh + multiply
    clamped = tl.minimum(tl.maximum(act_vals, 0.0), 6.0)
    result = clamped * acc

    # Store
    out_addrs = output_ptr + offs_n[None, :] * HW + offs_m[:, None]
    tl.store(out_addrs, result, mask=mask_out)


# ==================== Multi-batch kernel (3D grid, no division) ====================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_S': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_S': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_S': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_S': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
    ],
    key=['HW', 'N'],
)
@triton.jit
def fused_conv1x1_multi_batch_kernel(
    input_ptr, weight_ptr, bias_ptr, act_ptr, output_ptr,
    HW, K, N, C_in_HW, C_out_HW,
    BLOCK_S: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (N_batch, spatial_tiles, channel_tiles)
    batch_idx = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_s = offs_s < HW
    mask_n = offs_n < N

    # Input: batch_idx * C_in_HW + k * HW + spatial
    input_base = input_ptr + batch_idx * C_in_HW
    input_addrs = input_base + offs_k[None, :] * HW + offs_s[:, None]
    mask_input = mask_s[:, None] & (offs_k[None, :] < K)
    input_tile = tl.load(input_addrs, mask=mask_input, other=0.0)

    # Weight [BLOCK_K, BLOCK_N]
    weight_addrs = weight_ptr + offs_n[None, :] * K + offs_k[:, None]
    mask_weight = (offs_k[:, None] < K) & mask_n[None, :]
    weight_tile = tl.load(weight_addrs, mask=mask_weight, other=0.0)

    # Matmul
    acc = tl.dot(input_tile, weight_tile)

    # Bias
    bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias_vals[None, :].to(tl.float32)

    # Activation
    act_base = act_ptr + batch_idx * C_out_HW
    act_addrs = act_base + offs_n[None, :] * HW + offs_s[:, None]
    mask_out = mask_s[:, None] & mask_n[None, :]
    act_vals = tl.load(act_addrs, mask=mask_out, other=0.0).to(tl.float32)

    # Hardtanh + multiply
    clamped = tl.minimum(tl.maximum(act_vals, 0.0), 6.0)
    result = clamped * acc

    # Store
    out_addrs = output_ptr + batch_idx * C_out_HW + offs_n[None, :] * HW + offs_s[:, None]
    tl.store(out_addrs, result, mask=mask_out)


@torch.fx.wrap
def fused_conv1x1_hardtanh_mul(in_0, in_1, in_2, in_3):
    # in_0 = bias [C_out], in_1 = weight [C_out, C_in, 1, 1]
    # in_2 = input [N, C_in, H, W], in_3 = activation [N, C_out, H, W]
    N_batch = in_2.shape[0]
    C_in = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    C_out = in_1.shape[0]

    K = C_in
    N = C_out
    HW = H * W
    C_in_HW = C_in * HW
    C_out_HW = C_out * HW

    output = torch.empty_like(in_3)

    if N_batch == 1:
        M = HW
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
        fused_conv1x1_single_batch_kernel[grid](
            in_2, in_1, in_0, in_3, output,
            M, K, N, HW,
        )
    else:
        grid = lambda META: (N_batch, triton.cdiv(HW, META['BLOCK_S']), triton.cdiv(N, META['BLOCK_N']))
        fused_conv1x1_multi_batch_kernel[grid](
            in_2, in_1, in_0, in_3, output,
            HW, K, N, C_in_HW, C_out_HW,
        )

    return output


def replacement_func():
    return fused_conv1x1_hardtanh_mul