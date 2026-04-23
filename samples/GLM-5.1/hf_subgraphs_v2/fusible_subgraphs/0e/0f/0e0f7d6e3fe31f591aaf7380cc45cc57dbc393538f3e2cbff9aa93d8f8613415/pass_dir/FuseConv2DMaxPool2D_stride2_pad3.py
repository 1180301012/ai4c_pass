import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (3, 3), (1, 1), 1)
    tmp_3 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1, "conv2d_stride2_pad3_maxpool")


# ---- Triton kernel for fused Conv2D(stride=2,pad=3,kernel=7x7) + MaxPool2D(kernel=3,stride=2,pad=1) ----

@triton.jit
def fused_conv2d_maxpool_kernel_strided(
    input_ptr, weight_ptr, output_ptr,
    N, C_in, H_in, W_in, C_out,
    H_out_conv, W_out_conv, H_out_pool, W_out_pool,
    conv_stride_h: tl.constexpr, conv_stride_w: tl.constexpr,
    conv_pad_h: tl.constexpr, conv_pad_w: tl.constexpr,
    conv_kH: tl.constexpr, conv_kW: tl.constexpr,
    pool_kH: tl.constexpr, pool_kW: tl.constexpr,
    pool_stride: tl.constexpr, pool_pad: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_CO: tl.constexpr,
    BLOCK_HP: tl.constexpr, BLOCK_WP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    co_off = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    hp_off = pid_h * BLOCK_HP + tl.arange(0, BLOCK_HP)
    wp_off = pid_w * BLOCK_WP + tl.arange(0, BLOCK_WP)

    mask_n = n_off < N
    mask_co = co_off < C_out
    mask_hp = hp_off < H_out_pool
    mask_wp = wp_off < W_out_pool

    # Initialize max accumulator to -inf
    # Shape: [BLOCK_N, BLOCK_CO, BLOCK_HP, BLOCK_WP]
    max_acc = tl.full([BLOCK_N, BLOCK_CO, BLOCK_HP, BLOCK_WP], float('-inf'), dtype=tl.float32)

    # For each position in the 3x3 pool window
    for ph in range(pool_kH):
        for pw in range(pool_kW):
            # Conv output spatial location corresponding to this pool window position
            conv_h = hp_off * pool_stride - pool_pad + ph  # hp*2 - 1 + ph
            conv_w = wp_off * pool_stride - pool_pad + pw  # wp*2 - 1 + pw

            # Check if conv location is valid (within conv output bounds)
            valid_conv_h = (conv_h >= 0) & (conv_h < H_out_conv)
            valid_conv_w = (conv_w >= 0) & (conv_w < W_out_conv)

            # Compute conv value at this location: sum over (c_in, kh, kw)
            # conv[n, co, conv_h, conv_w] = sum_{ci,kh,kw} weight[co, ci, kh, kw] * input[n, ci, conv_h*stride_h - pad_h + kh, conv_w*stride_w - pad_w + kw]
            conv_acc = tl.zeros([BLOCK_N, BLOCK_CO, BLOCK_HP, BLOCK_WP], dtype=tl.float32)

            # Iterate over input channels and kernel positions
            for ci in range(C_in):
                for kh in range(conv_kH):
                    for kw in range(conv_kW):
                        # Input spatial location
                        in_h = conv_h * conv_stride_h - conv_pad_h + kh
                        in_w = conv_w * conv_stride_w - conv_pad_w + kw

                        valid_in_h = (in_h >= 0) & (in_h < H_in)
                        valid_in_w = (in_w >= 0) & (in_w < W_in)

                        # Combined validity mask
                        valid = mask_n & mask_co & mask_hp & mask_wp & valid_conv_h & valid_conv_w & valid_in_h & valid_in_w

                        # Load input: input[n, ci, in_h, in_w]
                        # Need to broadcast in_h (shape [BLOCK_HP]) and in_w (shape [BLOCK_WP]) to 4D
                        in_idx = n_off[:, None, None, None] * (C_in * H_in * W_in) + \
                                 ci * (H_in * W_in) + \
                                 in_h[None, None, :, None] * W_in + \
                                 in_w[None, None, None, :]

                        inp_val = tl.load(input_ptr + in_idx, mask=valid, other=0.0)

                        # Load weight: weight[co, ci, kh, kw]
                        w_idx = co_off[None, :, None, None] * (C_in * conv_kH * conv_kW) + \
                                ci * (conv_kH * conv_kW) + \
                                kh * conv_kW + kw

                        w_val = tl.load(weight_ptr + w_idx, mask=mask_co[None, :, None, None], other=0.0)

                        conv_acc += inp_val * w_val

            # Update max: where conv location is valid, consider it for max pooling
            valid_pool = mask_n & mask_co & mask_hp & mask_wp & valid_conv_h & valid_conv_w
            max_acc = tl.where(valid_pool & (conv_acc > max_acc), conv_acc, max_acc)

    # Cast to output dtype and store
    out_idx = n_off[:, None, None, None] * (C_out * H_out_pool * W_out_pool) + \
              co_off[None, :, None, None] * (H_out_pool * W_out_pool) + \
              hp_off[None, None, :, None] * W_out_pool + \
              wp_off[None, None, None, :]

    out_mask = mask_n & mask_co & mask_hp & mask_wp
    tl.store(output_ptr + out_idx, max_acc, mask=out_mask)


@torch.fx.wrap
def fused_conv2d_maxpool_dispatch(weight_tensor, input_tensor, route):
    if route == "conv2d_stride2_pad3_maxpool":
        return _fused_conv2d_stride2_pad3_maxpool(weight_tensor, input_tensor)
    elif route == "conv2d_stride1_pad1_maxpool":
        return _fused_conv2d_stride1_pad1_maxpool(weight_tensor, input_tensor)
    else:
        raise ValueError(f"Unknown route: {route}")


def _fused_conv2d_stride2_pad3_maxpool(weight_tensor, input_tensor):
    # Conv2d: stride=(2,2), padding=(3,3), dilation=(1,1), groups=1, kernel=7x7
    # MaxPool2d: kernel_size=3, stride=2, padding=1, dilation=1
    N, C_in, H_in, W_in = input_tensor.shape
    C_out = weight_tensor.shape[0]
    kH, kW = 7, 7

    H_out_conv = (H_in + 2 * 3 - 1 * (kH - 1) - 1) // 2 + 1
    W_out_conv = (W_in + 2 * 3 - 1 * (kW - 1) - 1) // 2 + 1
    H_out_pool = (H_out_conv + 2 * 1 - 3) // 2 + 1
    W_out_pool = (W_out_conv + 2 * 1 - 3) // 2 + 1

    output = torch.empty((N, C_out, H_out_pool, W_out_pool), dtype=input_tensor.dtype, device=input_tensor.device)

    BLOCK_N = 1
    BLOCK_CO = 8
    BLOCK_HP = 8
    BLOCK_WP = 8

    grid = (
        (N + BLOCK_N - 1) // BLOCK_N,
        (C_out + BLOCK_CO - 1) // BLOCK_CO,
        (H_out_pool + BLOCK_HP - 1) // BLOCK_HP,
        (W_out_pool + BLOCK_WP - 1) // BLOCK_WP,
    )

    fused_conv2d_maxpool_kernel_strided[grid](
        input_ptr=input_tensor, weight_ptr=weight_tensor, output_ptr=output,
        N=N, C_in=C_in, H_in=H_in, W_in=W_in, C_out=C_out,
        H_out_conv=H_out_conv, W_out_conv=W_out_conv,
        H_out_pool=H_out_pool, W_out_pool=W_out_pool,
        conv_stride_h=2, conv_stride_w=2, conv_pad_h=3, conv_pad_w=3,
        conv_kH=7, conv_kW=7,
        pool_kH=3, pool_kW=3, pool_stride=2, pool_pad=1,
        BLOCK_N=BLOCK_N, BLOCK_CO=BLOCK_CO, BLOCK_HP=BLOCK_HP, BLOCK_WP=BLOCK_WP,
    )

    return output


def _fused_conv2d_stride1_pad1_maxpool(weight_tensor, input_tensor):
    # Conv2d: stride=(1,1), padding=(1,1), dilation=(1,1), groups=1, kernel=3x3
    # MaxPool2d: kernel_size=3, stride=2, padding=1, dilation=1
    N, C_in, H_in, W_in = input_tensor.shape
    C_out = weight_tensor.shape[0]
    kH, kW = 3, 3

    H_out_conv = (H_in + 2 * 1 - 1 * (kH - 1) - 1) // 1 + 1
    W_out_conv = (W_in + 2 * 1 - 1 * (kW - 1) - 1) // 1 + 1
    H_out_pool = (H_out_conv + 2 * 1 - 3) // 2 + 1
    W_out_pool = (W_out_conv + 2 * 1 - 3) // 2 + 1

    output = torch.empty((N, C_out, H_out_pool, W_out_pool), dtype=input_tensor.dtype, device=input_tensor.device)

    BLOCK_N = 1
    BLOCK_CO = 8
    BLOCK_HP = 8
    BLOCK_WP = 8

    grid = (
        (N + BLOCK_N - 1) // BLOCK_N,
        (C_out + BLOCK_CO - 1) // BLOCK_CO,
        (H_out_pool + BLOCK_HP - 1) // BLOCK_HP,
        (W_out_pool + BLOCK_WP - 1) // BLOCK_WP,
    )

    fused_conv2d_maxpool_kernel_strided[grid](
        input_ptr=input_tensor, weight_ptr=weight_tensor, output_ptr=output,
        N=N, C_in=C_in, H_in=H_in, W_in=W_in, C_out=C_out,
        H_out_conv=H_out_conv, W_out_conv=W_out_conv,
        H_out_pool=H_out_pool, W_out_pool=W_out_pool,
        conv_stride_h=1, conv_stride_w=1, conv_pad_h=1, conv_pad_w=1,
        conv_kH=3, conv_kW=3,
        pool_kH=3, pool_kW=3, pool_stride=2, pool_pad=1,
        BLOCK_N=BLOCK_N, BLOCK_CO=BLOCK_CO, BLOCK_HP=BLOCK_HP, BLOCK_WP=BLOCK_WP,
    )

    return output


def replacement_func():
    return fused_conv2d_maxpool_dispatch