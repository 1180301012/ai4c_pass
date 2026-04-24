"""
Fused 3x3 Conv2d (stride=1, pad=1) + MaxPool2d (kernel=3, stride=2, pad=1, dil=1)
Pattern: resnetv2_18d intermediate block
"""
import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['N', 'C_out', 'H_out', 'W_out'],
)
@triton.jit
def _conv3x3_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    """
    3x3 Conv2d (stride=1, pad=1, dil=1, groups=1).
    Input layout: NCHW. Weight layout: [C_out, C_in, 3, 3].
    Output: [N, C_out, H_out, W_out] = same spatial as input (pad=1, stride=1).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C_out * H_out * W_out
    mask = offsets < total

    # Decode flat index -> (n, c_out, h_out, w_out)
    w_out = offsets % W_out
    h_out = (offsets // W_out) % H_out
    c_out = (offsets // (W_out * H_out)) % C_out
    n     = offsets // (W_out * H_out * C_out)

    # Accumulate 3x3 convolution
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for ic in range(3):
        for kh in range(3):
            for kw in range(3):
                ih = h_out + (kh - 1)   # pad=1, stride=1, dil=1
                iw = w_out + (kw - 1)   # pad=1, stride=1, dil=1

                valid = mask & (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in)

                in_idx = n * (C_in * H_in * W_in) + ic * (H_in * W_in) + ih * W_in + iw
                x = tl.load(input_ptr + in_idx, mask=valid, other=0.0).to(tl.float32)

                # weight[c_out, ic, kh, kw] flat index
                w_idx = c_out * (9 * C_in) + ic * 9 + kh * 3 + kw
                w_val = tl.load(weight_ptr + w_idx).to(tl.float32)

                acc = acc + x * w_val

    # Store
    out_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out
    tl.store(output_ptr + out_idx, acc.to(output_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['N', 'C', 'H_out', 'W_out'],
)
@triton.jit
def _maxpool2d_3x3_s2p1_kernel(
    input_ptr,
    output_ptr,
    N, C, H_in, W_in,
    H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    """
    2D MaxPool2d: kernel=3, stride=2, pad=1, dilation=1.
    Input: [N, C, H_in, W_in], Output: [N, C, H_out, W_out].
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * H_out * W_out
    mask = offsets < total

    # Decode flat index -> (n, c, h_out, w_out)
    w_out = offsets % W_out
    h_out = (offsets // W_out) % H_out
    c     = (offsets // (W_out * H_out)) % C
    n     = offsets // (W_out * H_out * C)

    # Max pool over 3x3 window with stride=2, pad=1
    NEG_INF = -1e9
    max_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32) + NEG_INF

    for th in range(3):
        for tw in range(3):
            h_in = h_out * 2 + th - 1   # stride=2, pad=1
            w_in = w_out * 2 + tw - 1

            valid = mask & (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)

            in_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in
            val = tl.load(input_ptr + in_idx, mask=valid, other=-1e9).to(tl.float32)

            max_val = tl.maximum(max_val, val)

    out_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out
    tl.store(output_ptr + out_idx, max_val.to(output_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def fused_conv3x3_stride1_maxpool(input_tensor, weight_tensor):
    # input_tensor: [N, C_in, H_in, W_in]
    # weight_tensor: [C_out, C_in, 3, 3]
    N, C_in, H_in, W_in = input_tensor.shape
    C_out = weight_tensor.shape[0]

    pad_h, pad_w = 1, 1
    H_out = H_in + 2 * pad_h - 2   # (H_in + 2*1 - 3)//1 + 1 = H_in (same spatial)
    W_out = W_in + 2 * pad_w - 2

    device = input_tensor.device
    dtype  = input_tensor.dtype

    weight = weight_tensor.to(device).contiguous()
    x = input_tensor.contiguous()

    # Step 1: Conv2d (3x3, stride=1, pad=1)
    conv_out = torch.empty((N, C_out, H_out, W_out), dtype=dtype, device=device)

    total_conv = N * C_out * H_out * W_out
    grid_conv = lambda meta: (triton.cdiv(total_conv, meta['BLOCK_SIZE']),)

    _conv3x3_kernel[grid_conv](
        x, weight, conv_out,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
    )

    # Step 2: MaxPool2d (3x3, stride=2, pad=1)
    output = torch.empty((N, C_out, (H_out + 1) // 2, (W_out + 1) // 2), dtype=dtype, device=device)

    N_batch = N
    C_out_pool = C_out
    H_out_pool = (H_out + 1) // 2
    W_out_pool = (W_out + 1) // 2
    total_pool = N_batch * C_out_pool * H_out_pool * W_out_pool
    grid_pool = lambda meta: (triton.cdiv(total_pool, meta['BLOCK_SIZE']),)

    _maxpool2d_3x3_s2p1_kernel[grid_pool](
        conv_out, output,
        N_batch, C_out_pool, H_out_pool, W_out_pool,
        H_out, W_out,
    )

    return (output,)


def replacement_func():
    return fused_conv3x3_stride1_maxpool