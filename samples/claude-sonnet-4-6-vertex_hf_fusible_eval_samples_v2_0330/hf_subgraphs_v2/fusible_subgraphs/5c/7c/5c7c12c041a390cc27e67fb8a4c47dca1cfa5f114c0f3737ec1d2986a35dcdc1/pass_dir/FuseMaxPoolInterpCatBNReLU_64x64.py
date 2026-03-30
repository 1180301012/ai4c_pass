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
def _fused_mp_cat_bn_relu_kernel_64(
    a_ptr,    # [N, C_a,   H,   W]  – cat_first (already at output resolution)
    b_ptr,    # [N, C_b, 2H, 2W]   – pool_input (will be max-pooled 2×2)
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
    B_W = 2 * W   # spatial width of pool_input

    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask    = hw_offsets < HW

    # BN parameters (always computed in fp32)
    mean = tl.load(mean_ptr   + pid_c).to(tl.float32)
    var  = tl.load(var_ptr    + pid_c).to(tl.float32)
    wt   = tl.load(weight_ptr + pid_c).to(tl.float32)
    bi   = tl.load(bias_ptr   + pid_c).to(tl.float32)
    scale = wt / tl.sqrt(var + eps)
    shift = bi - mean * scale

    is_from_a  = pid_c < C_a
    b_c_safe   = tl.where(is_from_a, 0, pid_c - C_a)

    # ── Load from A ───────────────────────────────────────────────────────────
    a_off = pid_n * C_a * HW + pid_c * HW + hw_offsets
    x_a   = tl.load(a_ptr + a_off, mask=hw_mask & is_from_a, other=0.0)

    # ── Load from B with 2×2 max-pool ────────────────────────────────────────
    h_out = hw_offsets // W        # output row  [0, H)
    w_out = hw_offsets % W         # output col  [0, W)
    bh    = 2 * h_out              # top-left input row
    bw    = 2 * w_out              # top-left input col

    b_ch_off = pid_n * C_b * (2 * H) * B_W + b_c_safe * (2 * H) * B_W
    not_a    = hw_mask & ~is_from_a

    v00 = tl.load(b_ptr + b_ch_off + bh       * B_W + bw,     mask=not_a, other=0.0)
    v01 = tl.load(b_ptr + b_ch_off + bh       * B_W + bw + 1, mask=not_a, other=0.0)
    v10 = tl.load(b_ptr + b_ch_off + (bh + 1) * B_W + bw,     mask=not_a, other=0.0)
    v11 = tl.load(b_ptr + b_ch_off + (bh + 1) * B_W + bw + 1, mask=not_a, other=0.0)
    x_b = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    x = tl.where(is_from_a, x_a, x_b)

    # BN + ReLU (fp32 accumulation)
    out_f32 = scale * x.to(tl.float32) + shift
    out_f32 = tl.maximum(out_f32, 0.0)
    out     = out_f32.to(x.dtype)

    tl.store(out_ptr + pid_n * C_total * HW + pid_c * HW + hw_offsets,
             out, mask=hw_mask)


# ── Pattern to match ──────────────────────────────────────────────────────────
# Matches: ERFNet_start73_end78_8
#   in_0 = running_mean, in_1 = running_var, in_2 = bias, in_3 = weight
#   in_4 = cat_first  [N, C_a,  64,  64]
#   in_5 = pool_input [N, C_b, 128, 128]
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = torch.nn.functional.max_pool2d(in_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, (64, 64), None, 'bilinear', False)
    tmp_6 = torch.cat([in_4, tmp_5], 1)
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@torch.fx.wrap
def _fused_maxpool_cat_bn_relu_64x64(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    in_0 = running_mean  [C_total]
    in_1 = running_var   [C_total]
    in_2 = bias          [C_total]
    in_3 = weight        [C_total]
    in_4 = cat_first     [N, C_a,  64,  64]
    in_5 = pool_input    [N, C_b, 128, 128]
    """
    N       = in_4.shape[0]
    C_a     = in_4.shape[1]
    C_b     = in_5.shape[1]
    H       = in_4.shape[2]
    W       = in_4.shape[3]
    C_total = C_a + C_b
    HW      = H * W

    out  = torch.empty((N, C_total, H, W), dtype=in_4.dtype, device=in_4.device)
    grid = lambda meta: (N, C_total, triton.cdiv(HW, meta['BLOCK_HW']))

    _fused_mp_cat_bn_relu_kernel_64[grid](
        in_4, in_5, out,
        in_0, in_1, in_3, in_2,   # mean, var, weight, bias
        C_a, C_b, H, W, 0.001,
    )
    return out


def replacement_func():
    return _fused_maxpool_cat_bn_relu_64x64