"""
Fused 7x7 Conv2d (stride=2, pad=3) + MaxPool2d (kernel=3, stride=2, pad=1, dil=1)
Pattern: resnetv2_101 stem branch
"""
import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (3, 3), (1, 1), 1)
    tmp_3 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return (tmp_3,)


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
def _fused_conv7x7_pool3x3_stride2_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    KH: tl.constexpr,
    KW: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    pool_kernel: tl.constexpr,
    pool_stride: tl.constexpr,
    pool_pad: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused 7x7 conv (stride=2, pad=3) + 3x3 max pool (stride=2, pad=1).
    Each program handles BLOCK_SIZE output elements (n, c_out, h_out, w_out).
    For each output element, computes the conv result for each of the
    pool_kernel x pool_kernel=9 positions and takes the max.
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

    # Load weight base address for each (n, c_out) pair
    # weight layout: [C_out, C_in, KH, KW] -> index: c_out * C_in * KH * KW + ic * KH * KW + kh * KW + kw
    KHW = KH * KW
    HW_in = H_in * W_in

    # Accumulator for each pool window position, initialized to a very small number
    # We'll iterate over pool window, computing conv values and taking max
    # The inner double-nested loop covers KH * KW conv multiplications for each pool position
    # We unroll over pool kernel positions (9 positions) and conv kernel (49 positions)

    # Instead of nested loops over pool positions (which are constexpr),
    # we directly compute the max across all pool window positions.
    # Conceptually: max_{th=0..2, tw=0..2} conv_out(h_out*2+th-1, w_out*2+tw-1)
    # where conv_out is the result of the 7x7 conv.
    # We compute this by iterating over (th, tw) and inside over (ic, kh, kw).

    # Use a large negative value as initial max
    NEG_INF = -1e9
    max_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32) + NEG_INF

    # Precompute the constant part of the weight base
    # weight[c_out, ic, kh, kw] at offset: c_out * (C_in * KHW) + ic * KHW + kh * KW + kw
    C_in_KHW = C_in * KHW
    # Base weight offset per c_out (per lane, c_out varies so we can't precompute fully)

    for th in range(pool_kernel):
        for tw in range(pool_kernel):
            # Input position after max pool: h_in_pos = h_out * pool_stride + th - pool_pad
            h_in_pos = h_out * pool_stride + th - pool_pad
            w_in_pos = w_out * pool_stride + tw - pool_pad

            # Check max pool boundary
            valid_h = (h_in_pos >= 0) & (h_in_pos < H_in)
            valid_w = (w_in_pos >= 0) & (w_in_pos < W_in)
            valid_pool = valid_h & valid_w & mask

            # 7x7 convolution accumulation for this pool position
            acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

            for ic in range(C_in):
                for kh in range(KH):
                    for kw in range(KW):
                        ih = h_in_pos * stride_h + kh - pad_h
                        iw = w_in_pos * stride_w + kw - pad_w

                        valid = valid_pool & (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in)

                        in_idx = n * (C_in * HW_in) + ic * HW_in + ih * W_in + iw
                        x = tl.load(input_ptr + in_idx, mask=valid, other=0.0).to(tl.float32)

                        # weight[c_out, ic, kh, kw]
                        w_idx = c_out * C_in_KHW + ic * KHW + kh * KW + kw
                        w_val = tl.load(weight_ptr + w_idx).to(tl.float32)

                        acc = acc + x * w_val

            # Update max_val
            max_val = tl.maximum(max_val, acc)

    # Store result (cast back to output dtype)
    out_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out
    tl.store(output_ptr + out_idx, max_val.to(output_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def fused_conv7x7_stride2_maxpool(input_tensor, weight_tensor):
    # input_tensor: [N, C_in, H_in, W_in]
    # weight_tensor: [C_out, C_in, KH, KW] = [64, 3, 7, 7]
    N, C_in, H_in, W_in = input_tensor.shape
    C_out = weight_tensor.shape[0]
    KH = weight_tensor.shape[2]
    KW = weight_tensor.shape[3]

    stride_h, stride_w = 2, 2
    pad_h,   pad_w   = 3, 3
    pool_k  = 3
    pool_sh, pool_sw = 2, 2
    pool_ph, pool_pw = 1, 1

    H_out = (H_in + 2 * pad_h - KH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1

    device = input_tensor.device
    dtype  = input_tensor.dtype

    # Move weights to same device as input if needed, ensure contiguous
    weight = weight_tensor.to(device).contiguous()
    x = input_tensor.contiguous()

    output = torch.empty((N, C_out, H_out, W_out), dtype=dtype, device=device)

    total = N * C_out * H_out * W_out
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)

    _fused_conv7x7_pool3x3_stride2_kernel[grid](
        x, weight, output,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        pool_k, pool_sh, pool_ph,
    )

    return (output,)


def replacement_func():
    return fused_conv7x7_stride2_maxpool