import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128}),
        triton.Config({'BLOCK_HW': 256}),
        triton.Config({'BLOCK_HW': 512}),
    ],
    key=['C_a', 'C_b', 'H', 'W'],
)
@triton.jit
def _fused_mp_cat_bn_relu_kernel_256(
    a_ptr,    # [N, C_a,   H,   W]  – cat_first (already at output resolution)
    b_ptr,    # [N, C_b, 2H, 2W]   – pool_input (max-pooled 2×2 in-kernel)
    out_ptr,  # [N, C_a+C_b, H, W]
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    C_a, C_b, H, W,
    eps,
    BLOCK_HW: tl.constexpr,
):
    pid_n  = tl.program_id(0)
    pid_c  = tl.program_id(1)
    pid_hw = tl.program_id(2)

    C_total = C_a + C_b
    HW  = H * W
    B_W = 2 * W   # width of pool_input

    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask    = hw_offsets < HW

    # BN parameters in fp32
    mean = tl.load(mean_ptr   + pid_c).to(tl.float32)
    var  = tl.load(var_ptr    + pid_c).to(tl.float32)
    wt   = tl.load(weight_ptr + pid_c).to(tl.float32)
    bi   = tl.load(bias_ptr   + pid_c).to(tl.float32)
    scale = wt / tl.sqrt(var + eps)
    shift = bi - mean * scale

    is_from_a = pid_c < C_a
    b_c_safe  = tl.where(is_from_a, 0, pid_c - C_a)

    # ── Load from A ───────────────────────────────────────────────────────────
    a_off = pid_n * C_a * HW + pid_c * HW + hw_offsets
    x_a   = tl.load(a_ptr + a_off, mask=hw_mask & is_from_a, other=0.0)

    # ── Load from B with 2×2 max-pool ────────────────────────────────────────
    h_out = hw_offsets // W
    w_out = hw_offsets % W
    bh    = 2 * h_out
    bw    = 2 * w_out

    b_ch_off = pid_n * C_b * (2 * H) * B_W + b_c_safe * (2 * H) * B_W
    not_a    = hw_mask & ~is_from_a

    v00 = tl.load(b_ptr + b_ch_off + bh       * B_W + bw,     mask=not_a, other=0.0)
    v01 = tl.load(b_ptr + b_ch_off + bh       * B_W + bw + 1, mask=not_a, other=0.0)
    v10 = tl.load(b_ptr + b_ch_off + (bh + 1) * B_W + bw,     mask=not_a, other=0.0)
    v11 = tl.load(b_ptr + b_ch_off + (bh + 1) * B_W + bw + 1, mask=not_a, other=0.0)
    x_b = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    x = tl.where(is_from_a, x_a, x_b)

    # BN + ReLU in fp32
    out_f32 = scale * x.to(tl.float32) + shift
    out_f32 = tl.maximum(out_f32, 0.0)
    out     = out_f32.to(x.dtype)

    tl.store(out_ptr + pid_n * C_total * HW + pid_c * HW + hw_offsets,
             out, mask=hw_mask)


# ── Pattern to match ──────────────────────────────────────────────────────────
# Matches: ERFNet_start1_end6_0
#   in_0 = pool_input  [N, C_b, 512, 512]
#   in_1 = running_mean, in_2 = running_var
#   in_3 = bias, in_4 = weight
#   in_5 = cat_first [N, C_a, 256, 256]
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_5 = torch.nn.functional.max_pool2d(in_0, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_6 = torch.nn.functional.interpolate(tmp_5, (256, 256), None, 'bilinear', False)
    tmp_7 = torch.cat([in_5, tmp_6], 1)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 0.001)
    tmp_9 = torch.nn.functional.relu(tmp_8, inplace=False)
    return tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@torch.fx.wrap
def _fused_maxpool_cat_bn_relu_256x256(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    in_0 = pool_input    [N, C_b, 512, 512]
    in_1 = running_mean  [C_total]
    in_2 = running_var   [C_total]
    in_3 = bias          [C_total]
    in_4 = weight        [C_total]
    in_5 = cat_first     [N, C_a, 256, 256]
    """
    N       = in_5.shape[0]
    C_a     = in_5.shape[1]
    C_b     = in_0.shape[1]
    H       = in_5.shape[2]
    W       = in_5.shape[3]
    C_total = C_a + C_b
    HW      = H * W

    out  = torch.empty((N, C_total, H, W), dtype=in_5.dtype, device=in_5.device)
    grid = lambda meta: (N, C_total, triton.cdiv(HW, meta['BLOCK_HW']))

    _fused_mp_cat_bn_relu_kernel_256[grid](
        in_5, in_0, out,
        in_1, in_2, in_4, in_3,   # mean, var, weight, bias
        C_a, C_b, H, W, 0.001,
    )
    return out


def replacement_func():
    return _fused_maxpool_cat_bn_relu_256x256