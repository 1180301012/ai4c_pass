import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Fused kernel: Conv2d(3x3, stride=1, pad=1) + MaxPool2d(3x3, stride=2, pad=1)
# For each max-pool output element we compute the conv output values on-the-fly
# inside the 3x3 pool window, avoiding the large intermediate tensor.
# -----------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 8},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64}, num_warps=8, num_stages=2),
    ],
    key=['C_in', 'C_out', 'H_out', 'W_out'],
)
@triton.jit
def _fused_conv3x3s1_maxpool3x3s2_kernel(
    input_ptr, weight_ptr, output_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    H_conv_out, W_conv_out,
    BLOCK_OC: tl.constexpr,
):
    """
    Grid: (N * H_out * W_out,  ceil(C_out / BLOCK_OC))
    Each program handles one spatial position and BLOCK_OC output channels.
    """
    pid_spatial = tl.program_id(0)
    pid_oc      = tl.program_id(1)

    HW_out = H_out * W_out
    n     = pid_spatial // HW_out
    hw    = pid_spatial % HW_out
    h_out = hw // W_out
    w_out = hw % W_out

    oc_base = pid_oc * BLOCK_OC
    oc_offs = oc_base + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < C_out

    # Accumulator initialised to -inf (max pooling identity)
    max_vals = tl.full((BLOCK_OC,), float('-inf'), dtype=tl.float32)

    # Pre-compute strides
    in_stride_n  = C_in * H_in * W_in
    in_stride_c  = H_in * W_in
    w_stride_oc  = C_in * 9   # C_in * 3 * 3

    # ---- Pool window loop: 3x3, stride=2, pad=1 ----
    for ph in range(3):
        for pw in range(3):
            conv_h = h_out * 2 + ph - 1
            conv_w = w_out * 2 + pw - 1

            valid_pool = (conv_h >= 0) & (conv_h < H_conv_out) & \
                         (conv_w >= 0) & (conv_w < W_conv_out)

            acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

            # ---- Convolution: 3x3, stride=1, pad=1 ----
            for ic in range(C_in):
                in_base = n * in_stride_n + ic * in_stride_c
                w_ic    = ic * 9  # ic * kH * kW

                for kh in tl.static_range(3):
                    for kw in tl.static_range(3):
                        ih = conv_h * 1 + kh - 1
                        iw = conv_w * 1 + kw - 1

                        valid_in = valid_pool & (ih >= 0) & (ih < H_in) & \
                                               (iw >= 0) & (iw < W_in)

                        ih_s = tl.maximum(0, tl.minimum(ih, H_in - 1))
                        iw_s = tl.maximum(0, tl.minimum(iw, W_in - 1))

                        in_offset = in_base + ih_s * W_in + iw_s
                        in_val = tl.load(input_ptr + in_offset,
                                         mask=valid_in, other=0.0).to(tl.float32)

                        w_kk      = w_ic + kh * 3 + kw
                        w_offsets = oc_offs * w_stride_oc + w_kk
                        w_vals    = tl.load(weight_ptr + w_offsets,
                                            mask=oc_mask, other=0.0).to(tl.float32)

                        acc = acc + in_val * w_vals

            # Update running max only for valid pool positions
            max_vals = tl.where(valid_pool, tl.maximum(max_vals, acc), max_vals)

    # ---- Write output [N, C_out, H_out, W_out] ----
    out_stride_n = C_out * H_out * W_out
    out_stride_c = H_out * W_out
    out_offsets  = (n * out_stride_n
                    + oc_offs * out_stride_c
                    + h_out * W_out
                    + w_out)
    tl.store(output_ptr + out_offsets,
             max_vals.to(output_ptr.dtype.element_ty),
             mask=oc_mask)


@torch.fx.wrap
def triton_fused_conv3x3s1p1_maxpool3x3s2p1(in_0, in_1):
    """
    in_0 : weight  [C_out, C_in, 3, 3]
    in_1 : input   [N, C_in, H_in, W_in]
    """
    device = in_1.device
    if device.type == 'cpu':
        in_0 = in_0.cuda()
        in_1 = in_1.cuda()
        device = in_1.device

    N, C_in, H_in, W_in = in_1.shape
    C_out = in_0.shape[0]

    # Conv output size (stride=1, padding=1, kernel=3, dilation=1)
    H_conv = (H_in + 2 * 1 - 3) // 1 + 1
    W_conv = (W_in + 2 * 1 - 3) // 1 + 1

    # Pool output size (stride=2, padding=1, kernel=3, dilation=1)
    H_out = (H_conv + 2 * 1 - 3) // 2 + 1
    W_out = (W_conv + 2 * 1 - 3) // 2 + 1

    output = torch.empty((N, C_out, H_out, W_out),
                          dtype=in_1.dtype, device=device)

    in_1_c = in_1.contiguous()
    in_0_c = in_0.contiguous()

    grid = lambda META: (
        N * H_out * W_out,
        triton.cdiv(C_out, META['BLOCK_OC']),
    )

    _fused_conv3x3s1_maxpool3x3s2_kernel[grid](
        in_1_c, in_0_c, output,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        H_conv, W_conv,
    )

    return (output,)


# -----------------------------------------------------------------------
# Pass interface
# -----------------------------------------------------------------------

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.nn.functional.max_pool2d(
        conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return triton_fused_conv3x3s1p1_maxpool3x3s2p1