import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: BN (inference) + ReLU + Cat
# Matches the sub-graph:
#   tmp_6 = batch_norm(tmp_5, mean, var, weight, bias, False, 0.1, 1e-05)
#   tmp_7 = relu(tmp_6, inplace=False)
#   tmp_8 = cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
# ---------------------------------------------------------------------------
def pattern(tmp_5, in_0, in_1, in_3, in_2, in_5, in_7, in_8, in_6):
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.relu(tmp_6, inplace=False)
    tmp_8 = torch.cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
    return tmp_8


def replacement_args(tmp_5, in_0, in_1, in_3, in_2, in_5, in_7, in_8, in_6):
    return (tmp_5, in_0, in_1, in_3, in_2, in_5, in_7, in_8, in_6)


# ---------------------------------------------------------------------------
# Triton kernel: fused BN + ReLU + Cat
#
# Layout: all inputs are contiguous [1, C, H, W] (NCHW).
# Output layout: [1, 5*C, H, W]
#   channels  0*C .. 1*C-1  ← in_5  (no BN, just copy)
#   channels  1*C .. 2*C-1  ← in_7
#   channels  2*C .. 3*C-1  ← in_8
#   channels  3*C .. 4*C-1  ← in_6
#   channels  4*C .. 5*C-1  ← conv output (tmp_5, with BN+ReLU)
#
# Grid: (5, ceil(HW / BLOCK_HW))
#   dim-0 = which of the 5 source tensors
#   dim-1 = spatial tile index
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def fused_bn_relu_cat_kernel(
    # 5 input tensor pointers (each [1, C, H, W] contiguous)
    in5_ptr, in7_ptr, in8_ptr, in6_ptr, conv_ptr,
    # BN parameters (each [C])
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    # Output [1, 5*C, H, W]
    out_ptr,
    # Scalar sizes
    C, HW,
    eps,
    # Compile-time dtype flags
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    src    = tl.program_id(0)   # which source: 0..4
    hw_pid = tl.program_id(1)   # spatial tile

    # Channel offset inside the source tensor (scalar, computed per source)
    ch = hw_pid * BLOCK_HW
    ch = tl.where(src == 0, 0 * C + ch, ch)
    ch = tl.where(src == 1, 1 * C + ch, ch)
    ch = tl.where(src == 2, 2 * C + ch, ch)
    ch = tl.where(src == 3, 3 * C + ch, ch)
    ch = tl.where(src == 4, 4 * C + ch, ch)

    # Spatial tile offsets (vector of length BLOCK_HW)
    sp_offsets = hw_pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask       = sp_offsets < HW

    # ---- Scalar boolean masks for each source ----
    # When src != this index the load returns other=0.0 (result discarded by tl.where)
    v_in5  = tl.load(in5_ptr  + ch * HW + sp_offsets, mask=mask & (src == 0), other=0.0)
    v_in7  = tl.load(in7_ptr  + ch * HW + sp_offsets, mask=mask & (src == 1), other=0.0)
    v_in8  = tl.load(in8_ptr  + ch * HW + sp_offsets, mask=mask & (src == 2), other=0.0)
    v_in6  = tl.load(in6_ptr  + ch * HW + sp_offsets, mask=mask & (src == 3), other=0.0)
    v_conv = tl.load(conv_ptr + ch * HW + sp_offsets, mask=mask & (src == 4), other=0.0)

    # Select the correct source value (scalar mask selects the whole vector)
    val = tl.where(src == 0, v_in5,
          tl.where(src == 1, v_in7,
          tl.where(src == 2, v_in8,
          tl.where(src == 3, v_in6, v_conv))))

    # ---- BN parameters for channel ch (scalar load, broadcast over spatial) ----
    mean_v   = tl.load(mean_ptr   + ch).to(tl.float32)
    var_v    = tl.load(var_ptr    + ch).to(tl.float32)
    weight_v = tl.load(weight_ptr + ch).to(tl.float32)
    bias_v   = tl.load(bias_ptr   + ch).to(tl.float32)

    # ---- BN (inference) + ReLU ----
    val_f32    = val.to(tl.float32)
    normalized = (val_f32 - mean_v) * tl.rsqrt(var_v + eps)
    val_bn     = normalized * weight_v + bias_v
    val_relu   = tl.maximum(val_bn, 0.0)

    # ---- Write output ----
    # Output channel = src * C + ch
    out_c      = src * C + ch
    out_offsets = out_c * HW + sp_offsets

    if IS_FP16:
        tl.store(out_ptr + out_offsets, val_relu.to(tl.float16),  mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + out_offsets, val_relu.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + out_offsets, val_relu, mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper – @torch.fx.wrap required
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_bn_relu_cat(tmp_5, in_0, in_1, in_3, in_2, in_5, in_7, in_8, in_6):
    """
    tmp_5 : [1, C, H, W]  – conv output (used as the 5th cat input)
    in_0  : [C]           – running_mean
    in_1  : [C]           – running_var
    in_3  : [C]           – BN weight
    in_2  : [C]           – BN bias
    in_5, in_7, in_8, in_6 : [1, C, H, W]
    returns: [1, 5*C, H, W]
    """
    C  = tmp_5.shape[1]
    H  = tmp_5.shape[2]
    W  = tmp_5.shape[3]
    HW = H * W

    out = torch.empty((1, 5 * C, H, W), dtype=tmp_5.dtype, device=tmp_5.device)

    is_fp16 = (tmp_5.dtype == torch.float16)
    is_bf16 = (tmp_5.dtype == torch.bfloat16)

    grid = lambda meta: (5, triton.cdiv(HW, meta['BLOCK_HW']))

    fused_bn_relu_cat_kernel[grid](
        in_5, in_7, in_8, in_6, tmp_5,
        in_0, in_1, in_3, in_2,
        out,
        C, HW, 1e-5,
        is_fp16, is_bf16,
    )

    return out


def replacement_func():
    return fused_bn_relu_cat