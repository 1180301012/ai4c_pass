import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Pattern A: conv2d(in_6, in_4) -> batch_norm(_, in_0, in_1, in_3, in_2) -> _ += in_5
    Matches deeppose_resnet_101_start96_end99_0
    """
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 += in_5
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # conv_input, conv_weight, running_mean, running_var, bn_weight, bn_bias, residual
    return (in_6, in_4, in_0, in_1, in_3, in_2, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['total_M', 'C_out', 'C_in'],
)
@triton.jit
def fused_conv1x1_bn_add_kernel_a(
    input_ptr, weight_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    residual_ptr, output_ptr,
    total_M, C_in, C_out, HW,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < total_M
    mask_n = offs_n < C_out

    batch_idx = offs_m // HW
    spatial_idx = offs_m % HW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < C_in

        # Load input [BLOCK_M, BLOCK_K]
        # input[n, c_in, hw] at offset: n * C_in * HW + c_in * HW + hw
        inp_ptrs = input_ptr + (batch_idx[:, None] * C_in * HW + offs_k[None, :] * HW + spatial_idx[:, None])
        inp = tl.load(inp_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load weight [BLOCK_K, BLOCK_N]
        # weight[c_out, c_in] at offset: c_out * C_in + c_in
        w_ptrs = weight_ptr + (offs_n[None, :] * C_in + offs_k[:, None])
        w = tl.load(w_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)

        acc += tl.dot(inp, w)

    # BN epilogue: output = (conv_out - mean) * gamma / sqrt(var + eps) + beta
    # Simplified: output = conv_out * scale + shift
    # where scale = gamma / sqrt(var + eps), shift = beta - mean * scale
    var = tl.load(var_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    scale = gamma * inv_std
    shift = beta - mean * scale

    result = acc * scale[None, :] + shift[None, :]

    # Add residual
    # residual[n, c_out, hw] at offset: n * C_out * HW + c_out * HW + hw
    res_ptrs = residual_ptr + (batch_idx[:, None] * C_out * HW + offs_n[None, :] * HW + spatial_idx[:, None])
    res = tl.load(res_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
    result += res

    # Store output
    out_ptrs = output_ptr + (batch_idx[:, None] * C_out * HW + offs_n[None, :] * HW + spatial_idx[:, None])
    tl.store(out_ptrs, result, mask=mask_m[:, None] & mask_n[None, :])


@torch.fx.wrap
def fused_conv1x1_bn_add_a(conv_input, conv_weight, running_mean, running_var, bn_weight, bn_bias, residual):
    batch_size = conv_input.shape[0]
    C_in = conv_input.shape[1]
    H = conv_input.shape[2]
    W = conv_input.shape[3]
    C_out = conv_weight.shape[0]
    HW = H * W
    total_M = batch_size * HW

    output = torch.empty_like(residual)

    grid = lambda META: (
        (total_M + META['BLOCK_M'] - 1) // META['BLOCK_M'],
        (C_out + META['BLOCK_N'] - 1) // META['BLOCK_N'],
    )

    fused_conv1x1_bn_add_kernel_a[grid](
        conv_input, conv_weight,
        running_mean, running_var, bn_weight, bn_bias,
        residual, output,
        total_M, C_in, C_out, HW,
    )

    return output


def replacement_func():
    return fused_conv1x1_bn_add_a