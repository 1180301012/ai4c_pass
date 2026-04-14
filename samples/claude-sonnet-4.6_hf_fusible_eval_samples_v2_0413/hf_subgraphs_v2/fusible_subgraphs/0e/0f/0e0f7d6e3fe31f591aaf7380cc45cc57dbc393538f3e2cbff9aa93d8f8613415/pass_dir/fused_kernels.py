"""
Shared Triton kernels for fused conv2d + max_pool2d.

Two patterns:
  route_7x7: conv(kH=7,kW=7,stride=2,pad=3,C_in=3) + maxpool(k=3,stride=2,pad=1)
  route_3x3: conv(kH=3,kW=3,stride=1,pad=1)          + maxpool(k=3,stride=2,pad=1)

Key optimization: avoid materialising the large intermediate conv output tensor by
computing conv and taking the max-pool reduction in a single fused kernel pass.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern 1 – 7×7 conv (stride=2, pad=3, C_in=3) + maxpool (k=3,stride=2,pad=1)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C_OUT': 64}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_C_OUT': 64}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_C_OUT': 32}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_C_OUT': 32}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_C_OUT': 16}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_C_OUT': 16}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_C_OUT': 64}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_C_OUT': 32}, num_warps=4,  num_stages=3),
    ],
    key=['N', 'H_in', 'W_in', 'C_out'],
)
@triton.jit
def _fused_conv7x7s2p3_maxpool3x3s2p1_kernel(
    input_ptr, weight_ptr, output_ptr,
    N, H_in, W_in,
    C_out, H_conv, W_conv, H_pool, W_pool,
    BLOCK_C_OUT: tl.constexpr,
):
    # Fixed parameters (hardcoded for this pattern):
    #   conv:  C_in=3, kH=7, kW=7, stride_h=stride_w=2, pad_h=pad_w=3
    #   pool:  kH=kW=3, stride=2, pad=1

    pid_spatial = tl.program_id(0)
    pid_c       = tl.program_id(1)

    # Decode (n, ph, pw) from the flattened spatial program id
    spatial_size = H_pool * W_pool
    n   = pid_spatial // spatial_size
    rem = pid_spatial %  spatial_size
    ph  = rem // W_pool
    pw  = rem %  W_pool

    # Output-channel slice this program handles
    c_out_start = pid_c * BLOCK_C_OUT
    c_out_offs  = c_out_start + tl.arange(0, BLOCK_C_OUT)
    c_out_mask  = c_out_offs < C_out

    # Running maximum – initialise to -∞
    NEG_INF   = -3.4028235e+38
    max_vals  = tl.full((BLOCK_C_OUT,), NEG_INF, dtype=tl.float32)

    # Base pointer for this batch element
    inp_n_base = n * (3 * H_in * W_in)

    # Pool window in conv-output space:  stride=2, pad=1
    h_conv_base = ph * 2 - 1
    w_conv_base = pw * 2 - 1

    # ---- pool window (3×3, unrolled at JIT time) ----
    for pkh in range(3):
        h_conv        = h_conv_base + pkh
        valid_h_conv  = (h_conv >= 0) & (h_conv < H_conv)

        for pkw in range(3):
            w_conv        = w_conv_base + pkw
            valid_w_conv  = (w_conv >= 0) & (w_conv < W_conv)

            # Compute conv2d at (n, :, h_conv, w_conv)
            conv_val   = tl.zeros((BLOCK_C_OUT,), dtype=tl.float32)
            h_in_base  = h_conv * 2 - 3   # conv stride=2, pad=3
            w_in_base  = w_conv * 2 - 3

            # ---- conv kernel (7×7, unrolled) ----
            for ckh in range(7):
                h_in       = h_in_base + ckh
                valid_h_in = (h_in >= 0) & (h_in < H_in)

                for ckw in range(7):
                    w_in       = w_in_base + ckw
                    valid_w_in = (w_in >= 0) & (w_in < W_in)
                    valid_in   = valid_h_in & valid_w_in

                    # ---- input-channel loop (C_in=3, unrolled) ----
                    for c_in in range(3):
                        inp_off = inp_n_base + c_in * (H_in * W_in) + h_in * W_in + w_in
                        inp_val = tl.load(input_ptr + inp_off,
                                          mask=valid_in, other=0.0).to(tl.float32)

                        # weight[c_out, c_in, ckh, ckw]  strides=[3*49, 49, 7, 1]
                        w_off  = c_out_offs * (3 * 49) + c_in * 49 + ckh * 7 + ckw
                        w_vals = tl.load(weight_ptr + w_off,
                                         mask=c_out_mask, other=0.0).to(tl.float32)
                        conv_val = conv_val + inp_val * w_vals

            # Update max; gate out-of-bounds pool positions with -∞
            valid_pos  = valid_h_conv & valid_w_conv
            neg_inf_v  = tl.full((BLOCK_C_OUT,), NEG_INF, dtype=tl.float32)
            gated_val  = tl.where(valid_pos, conv_val, neg_inf_v)
            max_vals   = tl.maximum(max_vals, gated_val)

    # Write pooled result
    out_off = (n * (C_out * H_pool * W_pool) +
               c_out_offs * (H_pool * W_pool) +
               ph * W_pool + pw)
    tl.store(output_ptr + out_off,
             max_vals.to(output_ptr.dtype.element_ty),
             mask=c_out_mask)


# ---------------------------------------------------------------------------
# Pattern 2 – 3×3 conv (stride=1, pad=1) + maxpool (k=3, stride=2, pad=1)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C_OUT': 64}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_C_OUT': 64}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_C_OUT': 32}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_C_OUT': 32}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_C_OUT': 16}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_C_OUT': 16}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_C_OUT': 64}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_C_OUT': 32}, num_warps=4,  num_stages=3),
    ],
    key=['N', 'C_in', 'H_in', 'W_in', 'C_out'],
)
@triton.jit
def _fused_conv3x3s1p1_maxpool3x3s2p1_kernel(
    input_ptr, weight_ptr, output_ptr,
    N, C_in, H_in, W_in,
    C_out, H_pool, W_pool,
    BLOCK_C_OUT: tl.constexpr,
):
    # Fixed parameters (hardcoded for this pattern):
    #   conv:  kH=3, kW=3, stride=1, pad=1  → H_conv = H_in, W_conv = W_in
    #   pool:  kH=kW=3, stride=2, pad=1

    pid_spatial = tl.program_id(0)
    pid_c       = tl.program_id(1)

    spatial_size = H_pool * W_pool
    n   = pid_spatial // spatial_size
    rem = pid_spatial %  spatial_size
    ph  = rem // W_pool
    pw  = rem %  W_pool

    c_out_start = pid_c * BLOCK_C_OUT
    c_out_offs  = c_out_start + tl.arange(0, BLOCK_C_OUT)
    c_out_mask  = c_out_offs < C_out

    NEG_INF   = -3.4028235e+38
    max_vals  = tl.full((BLOCK_C_OUT,), NEG_INF, dtype=tl.float32)

    inp_n_base = n * (C_in * H_in * W_in)

    # Pool window:  stride=2, pad=1
    h_conv_base = ph * 2 - 1
    w_conv_base = pw * 2 - 1

    for pkh in range(3):
        h_conv       = h_conv_base + pkh
        valid_h_conv = (h_conv >= 0) & (h_conv < H_in)   # H_conv == H_in

        for pkw in range(3):
            w_conv       = w_conv_base + pkw
            valid_w_conv = (w_conv >= 0) & (w_conv < W_in)   # W_conv == W_in

            conv_val  = tl.zeros((BLOCK_C_OUT,), dtype=tl.float32)
            h_in_base = h_conv - 1    # conv stride=1, pad=1
            w_in_base = w_conv - 1

            for ckh in range(3):
                h_in       = h_in_base + ckh
                valid_h_in = (h_in >= 0) & (h_in < H_in)

                for ckw in range(3):
                    w_in       = w_in_base + ckw
                    valid_w_in = (w_in >= 0) & (w_in < W_in)
                    valid_in   = valid_h_in & valid_w_in

                    # ---- input-channel loop (runtime, not unrolled) ----
                    for c_in in range(C_in):
                        inp_off = inp_n_base + c_in * (H_in * W_in) + h_in * W_in + w_in
                        inp_val = tl.load(input_ptr + inp_off,
                                          mask=valid_in, other=0.0).to(tl.float32)

                        # weight[c_out, c_in, ckh, ckw]  strides=[C_in*9, 9, 3, 1]
                        w_off  = c_out_offs * (C_in * 9) + c_in * 9 + ckh * 3 + ckw
                        w_vals = tl.load(weight_ptr + w_off,
                                         mask=c_out_mask, other=0.0).to(tl.float32)
                        conv_val = conv_val + inp_val * w_vals

            valid_pos = valid_h_conv & valid_w_conv
            neg_inf_v = tl.full((BLOCK_C_OUT,), NEG_INF, dtype=tl.float32)
            gated_val = tl.where(valid_pos, conv_val, neg_inf_v)
            max_vals  = tl.maximum(max_vals, gated_val)

    out_off = (n * (C_out * H_pool * W_pool) +
               c_out_offs * (H_pool * W_pool) +
               ph * W_pool + pw)
    tl.store(output_ptr + out_off,
             max_vals.to(output_ptr.dtype.element_ty),
             mask=c_out_mask)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper  (returned by replacement_func() in every pass file)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_conv_maxpool_wrapper(weight, input_tensor, route):
    """
    Unified entry point for both fused conv+maxpool patterns.

    Args:
        weight        – convolution weight tensor  [C_out, C_in, kH, kW]
        input_tensor  – input activation tensor    [N, C_in, H_in, W_in]
        route         – "route_7x7" or "route_3x3"

    Returns:
        output tensor  [N, C_out, H_pool, W_pool]
    """
    N, C_in, H_in, W_in = input_tensor.shape
    C_out = weight.shape[0]

    if route == "route_7x7":
        # conv: kH=7, kW=7, stride=2, pad=3, dil=1
        H_conv = (H_in + 2 * 3 - 1 * (7 - 1) - 1) // 2 + 1
        W_conv = (W_in + 2 * 3 - 1 * (7 - 1) - 1) // 2 + 1
        # pool: k=3, stride=2, pad=1, dil=1
        H_pool = (H_conv + 2 * 1 - 1 * (3 - 1) - 1) // 2 + 1
        W_pool = (W_conv + 2 * 1 - 1 * (3 - 1) - 1) // 2 + 1

        output = torch.empty((N, C_out, H_pool, W_pool),
                             dtype=input_tensor.dtype,
                             device=input_tensor.device)

        grid = lambda meta: (
            N * H_pool * W_pool,
            triton.cdiv(C_out, meta['BLOCK_C_OUT']),
        )
        _fused_conv7x7s2p3_maxpool3x3s2p1_kernel[grid](
            input_tensor, weight, output,
            N, H_in, W_in,
            C_out, H_conv, W_conv, H_pool, W_pool,
        )

    elif route == "route_3x3":
        # conv: kH=3, kW=3, stride=1, pad=1 → H_conv = H_in, W_conv = W_in
        # pool: k=3, stride=2, pad=1, dil=1
        H_pool = (H_in + 2 * 1 - 1 * (3 - 1) - 1) // 2 + 1
        W_pool = (W_in + 2 * 1 - 1 * (3 - 1) - 1) // 2 + 1

        output = torch.empty((N, C_out, H_pool, W_pool),
                             dtype=input_tensor.dtype,
                             device=input_tensor.device)

        grid = lambda meta: (
            N * H_pool * W_pool,
            triton.cdiv(C_out, meta['BLOCK_C_OUT']),
        )
        _fused_conv3x3s1p1_maxpool3x3s2p1_kernel[grid](
            input_tensor, weight, output,
            N, C_in, H_in, W_in,
            C_out, H_pool, W_pool,
        )

    else:
        # Should never be reached
        output = torch.empty(0, dtype=input_tensor.dtype, device=input_tensor.device)

    return output