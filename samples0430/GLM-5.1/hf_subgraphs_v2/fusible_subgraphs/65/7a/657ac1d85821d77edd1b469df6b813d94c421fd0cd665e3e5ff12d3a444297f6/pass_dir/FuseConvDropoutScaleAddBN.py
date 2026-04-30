import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
    conv2d = torch.conv2d(in_8, in_2, in_1, (1, 1), (0, 0), (1, 1), 1)
    tmp_8 = torch.nn.functional.dropout(conv2d, 0.0, False, False)
    tmp_9 = tmp_8 * in_0
    tmp_10 = in_7 + tmp_9
    tmp_11 = torch.nn.functional.batch_norm(tmp_10, in_3, in_4, in_6, in_5, False, 0.1, 1e-05)
    return (tmp_11, tmp_10)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['B', 'C_in', 'C_out', 'H', 'W'],
)
@triton.jit
def fused_conv_scale_add_bn_kernel(
    input_ptr, weight_ptr, bias_ptr, gamma_ptr, residual_ptr,
    running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    bn_out_ptr, residual_out_ptr,
    B, C_in, C_out, H, W,
    input_stride_b, input_stride_cin, input_stride_h, input_stride_w,
    weight_stride_cout, weight_stride_cin,
    gamma_stride_c,
    residual_stride_b, residual_stride_cout, residual_stride_h, residual_stride_w,
    bn_out_stride_b, bn_out_stride_cout, bn_out_stride_h, bn_out_stride_w,
    res_out_stride_b, res_out_stride_cout, res_out_stride_h, res_out_stride_w,
    eps,
    IS_FLOAT16: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    HW = H * W
    b_idx = offs_m // HW
    hw_idx = offs_m % HW
    h_idx = hw_idx // W
    w_idx = hw_idx % W

    mask_m = offs_m < B * HW
    mask_n = offs_n < C_out

    # Accumulator for conv output - always float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over input channels (K dimension)
    for k_start in range(0, C_in, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load input tile: [BLOCK_M, BLOCK_K]
        # input[b, cin, h, w]
        input_ptrs = input_ptr + (
            b_idx[:, None] * input_stride_b
            + offs_k[None, :] * input_stride_cin
            + h_idx[:, None] * input_stride_h
            + w_idx[:, None] * input_stride_w
        )
        input_mask = mask_m[:, None] & (offs_k[None, :] < C_in)
        a = tl.load(input_ptrs, mask=input_mask, other=0.0)

        # Load weight tile: [BLOCK_K, BLOCK_N] (transposed access for dot)
        # weight[c_out, c_in] accessed as weight[k, n]
        w_ptrs = weight_ptr + (
            offs_k[:, None] * weight_stride_cin
            + offs_n[None, :] * weight_stride_cout
        )
        w_mask = (offs_k[:, None] < C_in) & (offs_n[None, :] < C_out)
        b = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Dot product: acc += a @ b (accumulates in float32)
        acc += tl.dot(a, b)

    # Add bias (broadcast over M dimension)
    bias_ptrs = bias_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    # Dropout is identity (p=0.0, training=False) - skip

    # Apply gamma (layer scale) - broadcast over M dimension
    gamma_ptrs = gamma_ptr + offs_n * gamma_stride_c
    gamma_vals = tl.load(gamma_ptrs, mask=mask_n, other=1.0).to(tl.float32)
    scaled = acc * gamma_vals[None, :]

    # Load residual
    res_ptrs = residual_ptr + (
        b_idx[:, None] * residual_stride_b
        + offs_n[None, :] * residual_stride_cout
        + h_idx[:, None] * residual_stride_h
        + w_idx[:, None] * residual_stride_w
    )
    res_mask = mask_m[:, None] & mask_n[None, :]
    res_vals = tl.load(res_ptrs, mask=res_mask, other=0.0).to(tl.float32)

    # Residual add
    res_out_vals = res_vals + scaled

    # Batch norm parameters (all shape [C_out])
    rm = tl.load(running_mean_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    rv = tl.load(running_var_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    bw = tl.load(bn_weight_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    bb = tl.load(bn_bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    # BN: y = bw * (x - rm) / sqrt(rv + eps) + bb
    # = (bw / sqrt(rv + eps)) * x + (bb - rm * bw / sqrt(rv + eps))
    inv_std = bw / tl.sqrt(rv + eps)
    bn_shift = bb - rm * inv_std

    bn_out_vals = inv_std[None, :] * res_out_vals + bn_shift[None, :]

    # Store outputs with dtype conversion
    bn_out_ptrs = bn_out_ptr + (
        b_idx[:, None] * bn_out_stride_b
        + offs_n[None, :] * bn_out_stride_cout
        + h_idx[:, None] * bn_out_stride_h
        + w_idx[:, None] * bn_out_stride_w
    )
    res_out_ptrs = residual_out_ptr + (
        b_idx[:, None] * res_out_stride_b
        + offs_n[None, :] * res_out_stride_cout
        + h_idx[:, None] * res_out_stride_h
        + w_idx[:, None] * res_out_stride_w
    )

    out_mask = mask_m[:, None] & mask_n[None, :]

    if IS_FLOAT16:
        tl.store(bn_out_ptrs, bn_out_vals.to(tl.float16), mask=out_mask)
        tl.store(res_out_ptrs, res_out_vals.to(tl.float16), mask=out_mask)
    elif IS_BFLOAT16:
        tl.store(bn_out_ptrs, bn_out_vals.to(tl.bfloat16), mask=out_mask)
        tl.store(res_out_ptrs, res_out_vals.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(bn_out_ptrs, bn_out_vals.to(tl.float32), mask=out_mask)
        tl.store(res_out_ptrs, res_out_vals.to(tl.float32), mask=out_mask)


@torch.fx.wrap
def fused_conv_dropout_scale_add_bn(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
    # in_0 = gamma [C_out, 1, 1]
    # in_1 = conv_bias [C_out]
    # in_2 = conv_weight [C_out, C_in, 1, 1]
    # in_3 = running_mean [C_out]
    # in_4 = running_var [C_out]
    # in_5 = bn_bias [C_out]
    # in_6 = bn_weight [C_out]
    # in_7 = residual [B, C_out, H, W]
    # in_8 = conv_input [B, C_in, H, W]

    B, C_in, H, W = in_8.shape
    C_out = in_2.shape[0]

    # Allocate outputs with same dtype as input
    bn_out = torch.empty(B, C_out, H, W, dtype=in_8.dtype, device=in_8.device)
    residual_out = torch.empty(B, C_out, H, W, dtype=in_8.dtype, device=in_8.device)

    M = B * H * W

    dtype = in_8.dtype
    IS_FLOAT16 = dtype == torch.float16
    IS_BFLOAT16 = dtype == torch.bfloat16

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(C_out, META['BLOCK_N']),
    )

    fused_conv_scale_add_bn_kernel[grid](
        in_8, in_2, in_1, in_0, in_7,
        in_3, in_4, in_6, in_5,
        bn_out, residual_out,
        B, C_in, C_out, H, W,
        in_8.stride()[0], in_8.stride()[1], in_8.stride()[2], in_8.stride()[3],
        in_2.stride()[0], in_2.stride()[1],
        in_0.stride()[0],
        in_7.stride()[0], in_7.stride()[1], in_7.stride()[2], in_7.stride()[3],
        bn_out.stride()[0], bn_out.stride()[1], bn_out.stride()[2], bn_out.stride()[3],
        residual_out.stride()[0], residual_out.stride()[1], residual_out.stride()[2], residual_out.stride()[3],
        1e-05,
        IS_FLOAT16=IS_FLOAT16,
        IS_BFLOAT16=IS_BFLOAT16,
    )

    return bn_out, residual_out


def replacement_func():
    return fused_conv_dropout_scale_add_bn