import torch
import triton
import triton.language as tl


def pattern(tmp_5, in_0, in_1, in_2, in_3, in_5, in_6, in_7, in_8):
    """
    Match: batch_norm + relu + cat pattern.

    tmp_5  : BN input          [1, 512, 64, 64]
    in_0   : running_mean      [512]
    in_1   : running_var       [512]
    in_2   : BN bias           [512]
    in_3   : BN weight         [512]
    in_5   : cat slot 0        [1, 2048, 64, 64]
    in_6   : cat slot 3        [1,  512, 64, 64]
    in_7   : cat slot 1        [1,  512, 64, 64]
    in_8   : cat slot 2        [1,  512, 64, 64]
    """
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.relu(tmp_6, inplace=False)
    tmp_8 = torch.cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
    return tmp_8


def replacement_args(tmp_5, in_0, in_1, in_2, in_3, in_5, in_6, in_7, in_8):
    return (tmp_5, in_0, in_1, in_2, in_3, in_5, in_6, in_7, in_8)


# ---------------------------------------------------------------------------
# Triton kernel
# Output layout (dim=1, contiguous):
#   channels [0    : 2048] ← in_5
#   channels [2048 : 2560] ← in_7
#   channels [2560 : 3072] ← in_8
#   channels [3072 : 3584] ← in_6
#   channels [3584 : 4096] ← BN+ReLU(tmp_5)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def fused_bn_relu_cat_kernel(
    # --- copy sources ---
    in5_ptr,      # [1, 2048, H, W]
    in7_ptr,      # [1,  512, H, W]
    in8_ptr,      # [1,  512, H, W]
    in6_ptr,      # [1,  512, H, W]
    # --- BN source ---
    bn_in_ptr,    # [1,  512, H, W]  (tmp_5)
    # --- BN params ---
    bn_mean_ptr,  # [512]  (running_mean = in_0)
    bn_var_ptr,   # [512]  (running_var  = in_1)
    bn_w_ptr,     # [512]  (weight       = in_3)
    bn_b_ptr,     # [512]  (bias         = in_2)
    # --- output ---
    out_ptr,      # [1, 4096, H, W]
    # --- meta ---
    HW: tl.constexpr,      # H * W = 64 * 64 = 4096
    BLOCK_HW: tl.constexpr,
):
    c      = tl.program_id(0)   # output channel, 0 … 4095
    hw_pid = tl.program_id(1)   # spatial tile index

    hw_off = hw_pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    # HW is divisible by all BLOCK_HW choices → no masking needed

    out_off = c * HW + hw_off

    if c < 2048:
        val = tl.load(in5_ptr + c * HW + hw_off)
        tl.store(out_ptr + out_off, val)

    elif c < 2560:
        val = tl.load(in7_ptr + (c - 2048) * HW + hw_off)
        tl.store(out_ptr + out_off, val)

    elif c < 3072:
        val = tl.load(in8_ptr + (c - 2560) * HW + hw_off)
        tl.store(out_ptr + out_off, val)

    elif c < 3584:
        val = tl.load(in6_ptr + (c - 3072) * HW + hw_off)
        tl.store(out_ptr + out_off, val)

    else:
        c_bn = c - 3584
        # Load BN params (scalar per channel) → upcast to f32
        mean_f = tl.load(bn_mean_ptr + c_bn).to(tl.float32)
        var_f  = tl.load(bn_var_ptr  + c_bn).to(tl.float32)
        w_f    = tl.load(bn_w_ptr    + c_bn).to(tl.float32)
        b_f    = tl.load(bn_b_ptr    + c_bn).to(tl.float32)

        inv_std = w_f / tl.sqrt(var_f + 1e-5)
        shift   = b_f - mean_f * inv_std

        x   = tl.load(bn_in_ptr + c_bn * HW + hw_off)
        y   = x.to(tl.float32) * inv_std + shift
        y   = tl.maximum(y, 0.0)
        tl.store(out_ptr + out_off, y.to(x.dtype))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_bn_relu_cat(tmp_5, in_0, in_1, in_2, in_3, in_5, in_6, in_7, in_8):
    """
    tmp_5 : [1, 512,  64, 64]  BN input
    in_0  : [512]  running_mean
    in_1  : [512]  running_var
    in_2  : [512]  BN bias
    in_3  : [512]  BN weight
    in_5  : [1, 2048, 64, 64]
    in_6  : [1,  512, 64, 64]
    in_7  : [1,  512, 64, 64]
    in_8  : [1,  512, 64, 64]
    returns: [1, 4096, 64, 64]
    """
    H, W   = 64, 64
    HW     = H * W        # 4096
    C_OUT  = 4096         # 2048 + 512 + 512 + 512 + 512

    out = torch.empty((1, C_OUT, H, W), dtype=tmp_5.dtype, device=tmp_5.device)

    grid = lambda meta: (C_OUT, HW // meta['BLOCK_HW'])

    fused_bn_relu_cat_kernel[grid](
        # copy sources (in order: in5, in7, in8, in6)
        in_5, in_7, in_8, in_6,
        # BN source
        tmp_5,
        # BN params: mean=in_0, var=in_1, weight=in_3, bias=in_2
        in_0, in_1, in_3, in_2,
        # output
        out,
        # meta
        HW=HW,
    )

    return out


def replacement_func():
    return fused_bn_relu_cat