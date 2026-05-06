"""
Shared Triton kernels for fused conv2d + max_pool2d.

Two variants:
  A) 7x7 conv, stride=(2,2), padding=(3,3)  (resnetv2_101 stems)
  B) 3x3 conv, stride=(1,1), padding=(1,1)  (resnetv2_18d stems)

Max pool for both: KH=3, KS=2, PH=1, PW=1, floor mode.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Variant A: 7x7 conv, stride=2, pad=3  (resnetv2_101)
# ---------------------------------------------------------------------------

@triton.jit
def _conv2d_7x7_s2_p3_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N, C_IN, H_IN, W_IN,
    C_OUT,
    H_OUT, W_OUT,
    BLOCK_HW: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
):
    """
    output[n, c_out, h_out, w_out] = sum_{ic,kh,kw} input[n,ic,h_in,h_in]
        * weight[c_out, ic, kh, kw]
    where h_in = h_out*2 + kh - 3, w_in = w_out*2 + kw - 3 (zero-pad outside)
    """
    pid_hw = tl.program_id(0)
    pid_oc = tl.program_id(1)

    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
    oc_offs = pid_oc * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)  # [BLOCK_C_OUT]

    hw_mask = hw_offs < H_OUT * W_OUT
    oc_mask = oc_offs < C_OUT

    h_out = hw_offs // W_OUT  # [BLOCK_HW]
    w_out = hw_offs % W_OUT   # [BLOCK_HW]

    acc = tl.zeros([BLOCK_HW, BLOCK_C_OUT], dtype=tl.float32)

    # Unrolled 7x7 conv: K = C_IN * 7 * 7 = 3 * 49 = 147
    for ic in range(3):
        c_in_base = ic * H_IN * W_IN
        for kh in range(7):
            ih      = h_out * 2 + kh - 3                  # may be < 0
            ih_ok   = (ih >= 0) & (ih < H_IN)
            h_in_v  = tl.where(ih_ok, ih, 0)
            for kw in range(7):
                iw   = w_out * 2 + kw - 3                 # may be < 0
                iw_ok = (iw >= 0) & (iw < W_IN)
                in_ok = ih_ok & iw_ok

                in_off_v = n_offs[:, None] * (C_IN * H_IN * W_IN) + \
                           ic * H_IN * W_IN + \
                           h_in_v[None, :] * W_IN + \
                           iw[None, :]
                inp_v = tl.load(
                    input_ptr + in_off_v,
                    mask=hw_mask[:, None] & in_ok[None, :],
                    other=0.0
                )  # [BLOCK_HW, 1]  (broadcast over C_OUT)

                # weight[c_out, ic, kh, kw] — contiguous in C_OUT for fixed k
                k_idx   = ic * 49 + kh * 7 + kw
                w_idx_v = oc_offs * (3 * 49) + k_idx       # [BLOCK_C_OUT]
                w_v     = tl.load(weight_ptr + w_idx_v, mask=oc_mask, other=0.0)

                # acc += inp_v (scalars broadcast) · w_v
                acc = tl.fma(inp_v, w_v[None, :], acc)

    out_offs = n_offs[:, None] * (C_OUT * H_OUT * W_OUT) + \
               oc_offs[None, :] * (H_OUT * W_OUT) + \
               h_out[None, :] * W_OUT + \
               w_out[None, :]
    tl.store(output_ptr + out_offs, acc.to(input_ptr.dtype.element_ty),
             mask=hw_mask[:, None] & oc_mask[None, :])


def _run_conv2d_7x7_s2_p3(in_1, in_0):
    """
    in_1: input  [N, C_IN, H_IN, W_IN]
    in_0: weight [C_OUT, C_IN, KH, KW]  (KH=KW=7)
    """
    N, C_IN, H_IN, W_IN = in_1.shape
    C_OUT = in_0.shape[0]

    H_OUT = (H_IN + 2 * 3 - 7) // 2 + 1  # (H_IN + 3) // 2
    W_OUT = (W_IN + 2 * 3 - 7) // 2 + 1  # (W_IN + 3) // 2

    n_offs = tl.arange(0, N)

    output = torch.empty((N, C_OUT, H_OUT, W_OUT),
                         device=in_1.device, dtype=in_1.dtype)

    BLOCK_HW   = 64
    BLOCK_C_OUT = 64
    num_hw     = (H_OUT * W_OUT + BLOCK_HW  - 1) // BLOCK_HW
    num_oc     = (C_OUT      + BLOCK_C_OUT - 1) // BLOCK_C_OUT

    _conv2d_7x7_s2_p3_kernel[(num_hw, num_oc)](
        in_1, in_0, output,
        N, C_IN, H_IN, W_IN, C_OUT, H_OUT, W_OUT,
        BLOCK_HW=BLOCK_HW, BLOCK_C_OUT=BLOCK_C_OUT,
    )
    return output


# ---------------------------------------------------------------------------
# Variant B: 3x3 conv, stride=1, pad=1  (resnetv2_18d)
# ---------------------------------------------------------------------------

@triton.jit
def _conv2d_3x3_s1_p1_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N, C_IN, H_IN, W_IN,
    C_OUT,
    H_OUT, W_OUT,
    BLOCK_HW: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
):
    """
    output[n, c_out, h_out, w_out] = sum_{ic,kh,kw} input[n,ic,h_in,h_in]
        * weight[c_out,ic,kh,kw]
    where h_in = h_out*1 + kh - 1, w_in = w_out*1 + kw - 1
    """
    pid_hw = tl.program_id(0)
    pid_oc = tl.program_id(1)

    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oc_offs = pid_oc * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)

    hw_mask = hw_offs < H_OUT * W_OUT
    oc_mask = oc_offs < C_OUT

    h_out = hw_offs // W_OUT
    w_out = hw_offs % W_OUT

    acc = tl.zeros([BLOCK_HW, BLOCK_C_OUT], dtype=tl.float32)

    # Unrolled 3x3 conv: K = C_IN * 3 * 3 = 32 * 9 = 288
    for ic in range(32):
        c_in_base = ic * H_IN * W_IN
        for kh in range(3):
            ih    = h_out + kh - 1
            ih_ok = (ih >= 0) & (ih < H_IN)
            h_in_v = tl.where(ih_ok, ih, 0)
            for kw in range(3):
                iw   = w_out + kw - 1
                iw_ok = (iw >= 0) & (iw < W_IN)
                in_ok = ih_ok & iw_ok

                in_off_v = n_offs[:, None] * (C_IN * H_IN * W_IN) + \
                           ic * H_IN * W_IN + \
                           h_in_v[None, :] * W_IN + \
                           iw[None, :]
                inp_v = tl.load(
                    input_ptr + in_off_v,
                    mask=hw_mask[:, None] & in_ok[None, :],
                    other=0.0
                )

                k_idx   = ic * 9 + kh * 3 + kw
                w_idx_v = oc_offs * (32 * 9) + k_idx
                w_v     = tl.load(weight_ptr + w_idx_v, mask=oc_mask, other=0.0)

                acc = tl.fma(inp_v, w_v[None, :], acc)

    out_offs = n_offs[:, None] * (C_OUT * H_OUT * W_OUT) + \
               oc_offs[None, :] * (H_OUT * W_OUT) + \
               h_out[None, :] * W_OUT + \
               w_out[None, :]
    tl.store(output_ptr + out_offs, acc.to(input_ptr.dtype.element_ty),
             mask=hw_mask[:, None] & oc_mask[None, :])


def _run_conv2d_3x3_s1_p1(in_1, in_0):
    """
    in_1: input  [N, C_IN, H_IN, W_IN]
    in_0: weight [C_OUT, C_IN, KH, KW]  (KH=KW=3)
    """
    N, C_IN, H_IN, W_IN = in_1.shape
    C_OUT = in_0.shape[0]

    H_OUT = H_IN   # stride=1, pad=1 → same spatial size
    W_OUT = W_IN

    n_offs = tl.arange(0, N)

    output = torch.empty((N, C_OUT, H_OUT, W_OUT),
                         device=in_1.device, dtype=in_1.dtype)

    BLOCK_HW    = 64
    BLOCK_C_OUT = 64
    num_hw      = (H_OUT * W_OUT + BLOCK_HW  - 1) // BLOCK_HW
    num_oc      = (C_OUT      + BLOCK_C_OUT - 1) // BLOCK_C_OUT

    _conv2d_3x3_s1_p1_kernel[(num_hw, num_oc)](
        in_1, in_0, output,
        N, C_IN, H_IN, W_IN, C_OUT, H_OUT, W_OUT,
        BLOCK_HW=BLOCK_HW, BLOCK_C_OUT=BLOCK_C_OUT,
    )
    return output


# ---------------------------------------------------------------------------
# Max-pool kernel  (KH=3, KS=2, PH=1, PW=1, floor)
# ---------------------------------------------------------------------------

@triton.jit
def _max_pool3x3_s2_p1_kernel(
    inp_ptr,
    out_ptr,
    H_IN, W_IN,
    H_OUT, W_OUT,
    BLOCK_HW: tl.constexpr,
):
    """
    For each output position (h_out, w_out):
      h_in_l = h_out*2 - 1,  h_in_c = h_out*2,  h_in_r = h_out*2 + 1
      similarly for w_out.
    All 9 positions (clamped to 0 on invalid) are loaded and the max is taken.
    """
    pid = tl.program_id(0)
    hw_offs = pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < H_OUT * W_OUT

    h_out = hw_offs // W_OUT
    w_out = hw_offs % W_OUT

    # Indices of the 3×3 pool window corners relative to center (2*h_out, 2*w_out)
    dh = tl.full([BLOCK_HW], 2, dtype=tl.int32)
    dw = tl.full([BLOCK_HW], 2, dtype=tl.int32)

    corners = tl.stack([dh - 1, dh, dh + 1])   # [3, BLOCK_HW]
    cws     = tl.stack([dw - 1, dw, dw + 1])   # [3, BLOCK_HW]

    h_in = h_out[:, None] * dh[None, :] + tl.arange(0, 3)[:, None] - 1  # [3, BLOCK_HW]
    w_in = w_out[:, None] * dw[None, :] + tl.arange(0, 3)[:, None] - 1  # [3, BLOCK_HW]

    valid = (h_in >= 0) & (h_in < H_IN) & \
            (w_in >= 0) & (w_in < W_IN)  # [3, BLOCK_HW]

    flat_in = h_in * W_IN + w_in   # [3, BLOCK_HW]

    # Load the 9 values and take the max along axis-0
    vals = tl.load(inp_ptr + flat_in, mask=valid, other=-1e9)
    result = tl.max(vals, axis=0)   # [BLOCK_HW]

    out_offs = h_out * W_OUT + w_out   # [BLOCK_HW]
    tl.store(out_ptr + out_offs, result.to(out_ptr.dtype.element_ty),
             mask=hw_mask)


@torch.fx.wrap
def fused_conv_maxpool(in_1, in_0, route):
    """
    Fused replacement for conv2d(in_1, in_0, None, stride/pad) followed by
    max_pool2d(kernel=3, stride=2, pad=1).

    in_1 : feature-map input   [N, C_IN, H_IN, W_IN]
    in_0 : convolution weight  [C_OUT, C_IN, KH, KW]
    route: "A" — 7×7 kernel, stride=2, pad=3
           "B" — 3×3 kernel, stride=1,  pad=1
    """
    if route == "A":
        conv_out = _run_conv2d_7x7_s2_p3(in_1, in_0)
    else:
        conv_out = _run_conv2d_3x3_s1_p1(in_1, in_0)

    N, C_OUT, H_OUT, W_OUT = conv_out.shape
    BLOCK_HW = 64
    num_hw   = (H_OUT * W_OUT + BLOCK_HW - 1) // BLOCK_HW

    maxpool_out = torch.empty((N, C_OUT, H_OUT, W_OUT),
                              device=conv_out.device, dtype=conv_out.dtype)

    _max_pool3x3_s2_p1_kernel[(num_hw,)](
        conv_out, maxpool_out,
        H_OUT, W_OUT,
        BLOCK_HW=BLOCK_HW,
    )
    return maxpool_out