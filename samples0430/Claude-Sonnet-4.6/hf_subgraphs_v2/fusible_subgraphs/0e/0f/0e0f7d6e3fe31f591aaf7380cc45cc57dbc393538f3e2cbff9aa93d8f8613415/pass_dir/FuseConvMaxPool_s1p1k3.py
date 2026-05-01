"""
Fused Conv2D (stride=1, pad=1, kernel=3x3) + MaxPool2D (k=3, s=2, p=1) pass.
Matches resnetv2_18d stem pattern.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 32}, num_warps=8, num_stages=2),
    ],
    key=['C_out', 'C_in', 'out_h', 'out_w'],
)
@triton.jit
def fused_conv3x3_s1_maxpool_kernel(
    input_ptr, weight_ptr, output_ptr,
    N, C_in, H, W,
    C_out, conv_oh, conv_ow,
    out_h, out_w,
    input_sn, input_sc, input_sh, input_sw,
    weight_sco, weight_sci, weight_skh, weight_skw,
    output_sn, output_sc, output_sh, output_sw,
    BLOCK_OW: tl.constexpr,
):
    """
    Grid: (N * C_out * out_h, cdiv(out_w, BLOCK_OW))
    Each program computes BLOCK_OW output positions along the width dimension
    for a fixed (n, c_out, oh) combination.
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Decode (n, cout, oh) from pid_row
    oh = pid_row % out_h
    tmp = pid_row // out_h
    cout = tmp % C_out
    n = tmp // C_out

    # Output width positions
    ow_start = pid_col * BLOCK_OW
    ow = ow_start + tl.arange(0, BLOCK_OW)
    ow_mask = ow < out_w

    # Initialize with very negative value (max pooling baseline)
    max_val = tl.full((BLOCK_OW,), -1e30, dtype=tl.float32)

    # Pool kernel: 3x3, stride=2, padding=1
    for pkh in tl.static_range(3):
        # conv output row for this pool kernel row
        coh = oh * 2 + pkh - 1
        conv_h_ok = (coh >= 0) & (coh < conv_oh)

        for pkw in tl.static_range(3):
            # conv output col for this pool kernel col (vector over BLOCK_OW)
            cow = ow * 2 + pkw - 1
            conv_ok = ow_mask & conv_h_ok & (cow >= 0) & (cow < conv_ow)

            # Compute conv2d at (n, cout, coh, cow)
            # Conv: kernel=3x3, stride=1, padding=1, dilation=1
            conv_val = tl.zeros((BLOCK_OW,), dtype=tl.float32)

            for ci in range(C_in):
                for kh in tl.static_range(3):
                    # input row: ih = coh * conv_stride_h + kh - conv_pad_h
                    ih = coh * 1 + kh - 1
                    in_h_ok = conv_h_ok & (ih >= 0) & (ih < H)

                    for kw in tl.static_range(3):
                        # input col: iw = cow * conv_stride_w + kw - conv_pad_w (vector)
                        iw = cow * 1 + kw - 1
                        in_ok = conv_ok & in_h_ok & (iw >= 0) & (iw < W)

                        # Load weight (scalar - same for all ow positions)
                        wt = tl.load(
                            weight_ptr + cout * weight_sco + ci * weight_sci
                            + kh * weight_skh + kw * weight_skw
                        ).to(tl.float32)

                        # Load input (vector - differs per ow position)
                        in_off = (n * input_sn + ci * input_sc
                                  + ih * input_sh + iw * input_sw)
                        inp = tl.load(input_ptr + in_off, mask=in_ok, other=0.0).to(tl.float32)

                        conv_val = conv_val + inp * wt

            # Update max
            new_max = tl.maximum(max_val, conv_val)
            max_val = tl.where(conv_ok, new_max, max_val)

    # Store output
    out_off = n * output_sn + cout * output_sc + oh * output_sh + ow * output_sw
    tl.store(output_ptr + out_off, max_val, mask=ow_mask)


@torch.fx.wrap
def fused_conv3x3_s1_maxpool(in_0, in_1):
    """
    in_0: weight tensor [C_out, C_in, 3, 3]
    in_1: input tensor  [N, C_in, H, W]
    conv stride=(1,1), padding=(1,1), dilation=(1,1), groups=1
    maxpool kernel=3, stride=2, padding=1, dilation=1
    """
    weight = in_0
    inp = in_1

    # Move to CUDA if needed
    if not weight.is_cuda:
        weight = weight.cuda()
    if not inp.is_cuda:
        inp = inp.cuda()

    weight = weight.contiguous()
    inp = inp.contiguous()

    N, C_in, H, W = inp.shape
    C_out = weight.shape[0]

    # Conv output size: stride=1, padding=1, kernel=3, dilation=1
    conv_oh = (H + 2 * 1 - 1 * (3 - 1) - 1) // 1 + 1  # = H
    conv_ow = (W + 2 * 1 - 1 * (3 - 1) - 1) // 1 + 1  # = W

    # MaxPool output size: kernel=3, stride=2, padding=1, dilation=1
    out_h = (conv_oh + 2 * 1 - 1 * (3 - 1) - 1) // 2 + 1
    out_w = (conv_ow + 2 * 1 - 1 * (3 - 1) - 1) // 2 + 1

    output = torch.empty((N, C_out, out_h, out_w), dtype=inp.dtype, device=inp.device)

    # Grid: rows = N * C_out * out_h, cols = ceil(out_w / BLOCK_OW)
    def grid(meta):
        return (N * C_out * out_h, triton.cdiv(out_w, meta['BLOCK_OW']))

    fused_conv3x3_s1_maxpool_kernel[grid](
        inp, weight, output,
        N, C_in, H, W,
        C_out, conv_oh, conv_ow,
        out_h, out_w,
        inp.stride(0), inp.stride(1), inp.stride(2), inp.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )

    return output


# ── Pattern to match ────────────────────────────────────────────────────────

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.nn.functional.max_pool2d(
        conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False
    )
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_conv3x3_s1_maxpool