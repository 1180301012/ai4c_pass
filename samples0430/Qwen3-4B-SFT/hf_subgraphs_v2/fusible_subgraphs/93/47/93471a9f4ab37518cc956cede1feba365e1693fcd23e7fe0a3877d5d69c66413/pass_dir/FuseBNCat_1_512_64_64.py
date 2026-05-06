import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: batch_norm(tmp_5, running_mean, running_var, weight, bias, False, …)
#          → relu(…)
#          → cat([in_5, in_7, in_8, in_6, relu_out], dim=1)
# ─────────────────────────────────────────────────────────────────────────────
def pattern(tmp_5, in_0, in_1, in_3, in_2, in_5, in_6, in_7, in_8):
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.relu(tmp_6, inplace=False)
    tmp_8 = torch.cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
    return tmp_8


def replacement_args(tmp_5, in_0, in_1, in_3, in_2, in_5, in_6, in_7, in_8):
    # Conv output is tmp_5 [1, 512, 64, 64]; BN params are in_0/in_1/in_3/in_2
    # Cat sources: in_5, in_7, in_8, in_6, tmp_7
    return (tmp_5, in_0, in_1, in_3, in_2, in_5, in_6, in_7, in_8)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused BN + ReLU + concatenation
#
# Output layout [1, C_total, H_out, W_out] where C_total = 2560.
#   channels 0–2047   ← in_5          [0: 2048]
#   channels 2048–2560 ← in_7         [2048:2560]
#   channels 2560–3072 ← in_8         [2560:3072]
#   channels 3072–3584 ← in_6         [3072:3584]
#   channels 3584–4096 ← relu(BN(tmp_5))[3584:4096]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['C_total', 'HW'],
)
@triton.jit
def _fused_bn_relu_cat_kernel(
    tmp5_ptr,        # [1, 512, 64, 64]  float ptr
    mean_ptr,        # [512]             float ptr
    var_ptr,         # [512]             float ptr
    bnw_ptr,         # [512]             float ptr   (BN weight γ)
    bnb_ptr,         # [512]             float ptr   (BN bias   β)
    in5_ptr,         # [1, 2048, 64, 64] float ptr
    in6_ptr,         # [1,  512, 64, 64] float ptr
    in7_ptr,         # [1,  512, 64, 64] float ptr
    in8_ptr,         # [1,  512, 64, 64] float ptr
    out_ptr,         # [1, 2560, 64, 64] float ptr
    C_total,         # 2560
    HW,              # 64*64 = 4096
    BLOCK_SIZE: tl.constexpr,
):
    pid       = tl.program_id(0)
    offsets   = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total     = C_total * HW
    mask      = offsets < total

    # ── Which channel is each output element in? ──────────────────────────
    ch_idx = (offsets // HW).to(tl.int32)

    # ── Segment masks ─────────────────────────────────────────────────────
    in_seg5 = ch_idx < 2048
    in_seg7 = (ch_idx >= 2048) & (ch_idx < 2560)
    in_seg8 = (ch_idx >= 2560) & (ch_idx < 3072)
    in_seg6 = (ch_idx >= 3072) & (ch_idx < 3584)
    in_segB = ch_idx >= 3584   # BN + ReLU output

    # ── Compute BN (fp32 for numerical stability), inclusive of ReLU ─────
    # BN: y = (x − mean) / sqrt(var + eps) * γ + β
    mean   = tl.load(mean_ptr + ch_idx, mask=mask & in_segB, other=-1.0).to(tl.float32)
    var    = tl.load(var_ptr  + ch_idx, mask=mask & in_segB, other=1.0).to(tl.float32)
    bnw    = tl.load(bnw_ptr  + ch_idx, mask=mask & in_segB, other=1.0).to(tl.float32)
    bnb    = tl.load(bnb_ptr  + ch_idx, mask=mask & in_segB, other=0.0).to(tl.float32)
    eps    = 1e-5
    inv_std = 1.0 / tl.sqrt(var + eps)
    bn_scale = bnw * inv_std
    bn_bias  = bnb - mean * bn_scale
    # Apply BN then ReLU in float32
    x_scaled = tl.load(tmp5_ptr + offsets, mask=mask & in_segB, other=0.0).to(tl.float32)
    y = tl.maximum(bn_scale * x_scaled + bn_bias, 0.0)

    # ── Load from the right source ─────────────────────────────────────────
    out_offsets = offsets % total          # same flat index (batch=1)

    src5 = tl.load(in5_ptr + out_offsets, mask=mask & in_seg5, other=0.0)
    src7 = tl.load(in7_ptr + out_offsets, mask=mask & in_seg7, other=0.0)
    src8 = tl.load(in8_ptr + out_offsets, mask=mask & in_seg8, other=0.0)
    src6 = tl.load(in6_ptr + out_offsets, mask=mask & in_seg6, other=0.0)
    srcB = y.to(out_ptr.dtype.element_ty)

    val = tl.where(in_seg5, src5,
          tl.where(in_seg7, src7,
          tl.where(in_seg8, src8,
          tl.where(in_seg6, src6, srcB))))

    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def fused_bn_relu_cat(tmp_5, in_0, in_1, in_3, in_2, in_5, in_6, in_7, in_8):
    """
    Fused batch_norm(relu(x)) + concatenation along dim=1.

    in_0  – running_mean   [512]
    in_1  – running_var    [512]
    in_3  – BN weight γ    [512]
    in_2  – BN bias   β    [512]
    in_5  – [1, 2048, 64, 64]
    in_6  – [1, 512,  64, 64]
    in_7  – [1, 512,  64, 64]
    in_8  – [1, 512,  64, 64]
    """
    C_total = 2560
    H_out   = 64
    W_out   = 64
    HW      = H_out * W_out   # 4096
    total   = C_total * HW    # 104_857_600

    out  = torch.empty((1, C_total, H_out, W_out),
                       dtype=tmp_5.dtype, device=tmp_5.device)

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)

    _fused_bn_relu_cat_kernel[grid](
        tmp_5,           # tmp5_ptr
        in_0,            # mean_ptr
        in_1,            # var_ptr
        in_3,            # bnw_ptr  (γ)
        in_2,            # bnb_ptr  (β)
        in_5,            # in5_ptr
        in_6,            # in6_ptr
        in_7,            # in7_ptr
        in_8,            # in8_ptr
        out,             # out_ptr
        C_total,
        HW,
    )

    return out


def replacement_func():
    return fused_bn_relu_cat