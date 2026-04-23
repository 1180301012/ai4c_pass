import torch
import triton
import triton.language as tl

# Pattern matching function - must match model.py exactly (without None cleanup statements)
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.leaky_relu(tmp_6, 0.01, True)
    tmp_8 = tmp_7 + in_5
    return (tmp_8,)

# Argument extraction - pass inputs needed for the fused kernel
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # in_0: running_mean [128], in_1: running_var [128], in_2: bias [128], in_3: weight [128]
    # in_4: conv_weight [128,64,3,3], in_5: residual [B,128,H,W], in_6: conv_input [B,64,H,W]
    return (in_6, in_4, in_5, in_0, in_1, in_3, in_2)


@triton.jit
def fused_bn_leaky_relu_add_kernel(
    conv_input_ptr, conv_weight_ptr, residual_ptr,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    output_ptr,
    N, C, H, W,
    stride_conv_input_n, stride_conv_input_c, stride_conv_input_h, stride_conv_input_w,
    stride_conv_weight_oc, stride_conv_weight_ic, stride_conv_weight_h, stride_conv_weight_w,
    stride_residual_n, stride_residual_c, stride_residual_h, stride_residual_w,
    stride_output_n, stride_output_c, stride_output_h, stride_output_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Each program instance computes a [BLOCK_M, BLOCK_N] tile of the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the output tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row indices (flattened N*H)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # column indices (W)

    # Total number of rows = N * H
    total_rows = N * H
    row_mask = offs_m < total_rows
    col_mask = offs_n < W

    # Decode row index into (n_idx, h_idx)
    n_idx = offs_m // H
    h_idx = offs_m % H
    n_mask = n_idx < N
    h_mask = h_idx < H

    # Output accumulator for each output channel (we iterate over output channels)
    # We'll compute one output channel at a time to keep register pressure low
    # Actually, let's compute all output channels for this spatial tile
    # For each output channel c_out:
    for c_out in range(C):
        # Load BN parameters for this channel
        bn_w = tl.load(bn_weight_ptr + c_out)
        bn_b = tl.load(bn_bias_ptr + c_out)
        bn_mean = tl.load(bn_mean_ptr + c_out)
        bn_var = tl.load(bn_var_ptr + c_out)

        # Precompute BN scale and shift in float32
        inv_var = 1.0 / tl.sqrt(bn_var + 1e-05)
        bn_scale = bn_w * inv_var
        bn_shift = bn_b - bn_w * bn_mean * inv_var

        # Accumulate conv output for this output channel and spatial tile
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Conv2d: weight shape [C_out=128, C_in=64, 3, 3], stride=1, padding=1
        # For each input channel and each kernel position
        for c_in in range(64):  # C_in = 64
            for kh in range(3):
                for kw in range(3):
                    # Compute input spatial position with padding=1
                    ih = h_idx + kh - 1  # padding=1
                    iw = offs_n + kw - 1  # padding=1

                    ih_mask = ih >= 0 and ih < H
                    iw_mask = iw >= 0 and iw < W
                    load_mask = row_mask & col_mask & ih_mask & iw_mask

                    # Compute input pointer offset
                    input_offs = n_idx * stride_conv_input_n + c_in * stride_conv_input_c + ih * stride_conv_input_h + iw * stride_conv_input_w

                    # Load input values (use 0 for out-of-bounds / padding)
                    x_val = tl.load(conv_input_ptr + input_offs, mask=load_mask, other=0.0)

                    # Load weight value
                    w_offs = c_out * stride_conv_weight_oc + c_in * stride_conv_weight_ic + kh * stride_conv_weight_h + kw * stride_conv_weight_w
                    w_val = tl.load(conv_weight_ptr + w_offs)

                    # Accumulate
                    acc += x_val * w_val

        # Apply BN: y = scale * x + shift
        acc = bn_scale * acc + bn_shift

        # Apply LeakyReLU with slope 0.01
        acc = tl.where(acc >= 0, acc, acc * 0.01)

        # Load residual and add
        residual_offs = n_idx * stride_residual_n + c_out * stride_residual_c + h_idx * stride_residual_h + offs_n * stride_residual_w
        res_mask = row_mask & col_mask
        res_val = tl.load(residual_ptr + residual_offs, mask=res_mask, other=0.0)
        acc = acc + res_val

        # Store output
        output_offs = n_idx * stride_output_n + c_out * stride_output_c + h_idx * stride_output_h + offs_n * stride_output_w
        tl.store(output_ptr + output_offs, acc, mask=res_mask)


@triton.jit
def fused_bn_leaky_relu_add_kernel_v2(
    conv_input_ptr, conv_weight_ptr, residual_ptr,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    output_ptr,
    N, C_in, C_out, H, W,
    stride_conv_input_n, stride_conv_input_c, stride_conv_input_h, stride_conv_input_w,
    stride_conv_weight_oc, stride_conv_weight_ic, stride_conv_weight_h, stride_conv_weight_w,
    stride_residual_n, stride_residual_c, stride_residual_h, stride_residual_w,
    stride_output_n, stride_output_c, stride_output_h, stride_output_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_out = pid_c

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    total_rows = N * H
    row_mask = offs_m < total_rows
    col_mask = offs_n < W

    n_idx = offs_m // H
    h_idx = offs_m % H

    # Load BN parameters for this channel
    bn_w = tl.load(bn_weight_ptr + c_out).to(tl.float32)
    bn_b = tl.load(bn_bias_ptr + c_out).to(tl.float32)
    bn_mean = tl.load(bn_mean_ptr + c_out).to(tl.float32)
    bn_var = tl.load(bn_var_ptr + c_out).to(tl.float32)

    inv_var = 1.0 / tl.sqrt(bn_var + 1e-05)
    bn_scale = bn_w * inv_var
    bn_shift = bn_b - bn_w * bn_mean * inv_var

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for c_in in range(C_in):
        for kh in range(3):
            for kw in range(3):
                ih = h_idx + kh - 1
                iw = offs_n + kw - 1

                ih_valid = (ih >= 0) & (ih < H)
                iw_valid = (iw >= 0) & (iw < W)
                load_mask = row_mask & col_mask & ih_valid & iw_valid

                input_offs = n_idx * stride_conv_input_n + c_in * stride_conv_input_c + ih * stride_conv_input_h + iw * stride_conv_input_w

                x_val = tl.load(conv_input_ptr + input_offs, mask=load_mask, other=0.0).to(tl.float32)

                w_val = tl.load(conv_weight_ptr + c_out * stride_conv_weight_oc + c_in * stride_conv_weight_ic + kh * stride_conv_weight_h + kw * stride_conv_weight_w).to(tl.float32)

                acc += x_val * w_val

    # BN
    acc = bn_scale * acc + bn_shift
    # LeakyReLU
    acc = tl.where(acc >= 0.0, acc, acc * 0.01)
    # Add residual
    residual_offs = n_idx * stride_residual_n + c_out * stride_residual_c + h_idx * stride_residual_h + offs_n * stride_residual_w
    res_mask = row_mask & col_mask
    res_val = tl.load(residual_ptr + residual_offs, mask=res_mask, other=0.0).to(tl.float32)
    acc = acc + res_val

    # Store
    output_dtype = output_ptr.dtype.element_ty
    output_offs = n_idx * stride_output_n + c_out * stride_output_c + h_idx * stride_output_h + offs_n * stride_output_w
    tl.store(output_ptr + output_offs, acc.to(output_dtype), mask=res_mask)


@torch.fx.wrap
def fused_bn_leaky_relu_add_conv2d(conv_input, conv_weight, residual, bn_mean, bn_var, bn_weight, bn_bias):
    # conv_input: [N, C_in, H, W], conv_weight: [C_out, C_in, 3, 3]
    # residual: [N, C_out, H, W]
    # bn_mean, bn_var, bn_weight, bn_bias: [C_out]
    N, C_in, H, W = conv_input.shape
    C_out = conv_weight.shape[0]

    output = torch.empty((N, C_out, H, W), dtype=conv_input.dtype, device=conv_input.device)

    # Grid: (N*H) rows, C_out channels
    BLOCK_M = 32
    BLOCK_N = 64

    num_rows = N * H
    grid_m = (num_rows + BLOCK_M - 1) // BLOCK_M
    grid_c = C_out

    grid = (grid_m, grid_c)

    fused_bn_leaky_relu_add_kernel_v2[grid](
        conv_input, conv_weight, residual,
        bn_mean, bn_var, bn_weight, bn_bias,
        output,
        N, C_in, C_out, H, W,
        conv_input.stride(0), conv_input.stride(1), conv_input.stride(2), conv_input.stride(3),
        conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2), conv_weight.stride(3),
        residual.stride(0), residual.stride(1), residual.stride(2), residual.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    return output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.leaky_relu(tmp_6, 0.01, True)
    tmp_8 = tmp_7 + in_5
    return (tmp_8,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_6, in_4, in_5, in_0, in_1, in_3, in_2)


def replacement_func():
    return fused_bn_leaky_relu_add_conv2d