"""
Pass: FuseCatBNReLU  (file kept as FusePoolCatBNReLU_64 for JSON reference)

Fuses: cat([a, b], 1) + batch_norm(inference) + relu  → single Triton kernel

Matches ALL three ERFNet subgraph variants:
  ERFNet_start73_end78_8  : cat([in_4, tmp_5],1) + BN(in_0,in_1,in_3,in_2) + relu
  ERFNet_start7_end12_1   : cat([in_4, tmp_5],1) + BN(in_0,in_1,in_3,in_2) + relu
  ERFNet_start1_end6_0    : cat([in_5, tmp_6],1) + BN(in_1,in_2,in_4,in_3) + relu

All three produce the same structural pattern when abstracting the cat inputs.
"""

import torch
import triton
import triton.language as tl


# ── Triton kernel ────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['C_A', 'C_B', 'HW'],
)
@triton.jit
def _fused_cat_bn_relu_kernel(
    tensor_a_ptr,   # [N, C_A, H, W]  – first tensor in cat
    tensor_b_ptr,   # [N, C_B, H, W]  – second tensor in cat (already at right HxW)
    mean_ptr,       # [C_total]
    var_ptr,        # [C_total]
    weight_ptr,     # [C_total]
    bias_ptr,       # [C_total]
    out_ptr,        # [N, C_total, H, W]
    N, C_A, C_B, HW, C_total,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: (N * C_total,  cdiv(HW, BLOCK_SIZE))
    nc       = tl.program_id(0)   # flat (n * C_total + c)
    hw_blk   = tl.program_id(1)

    c  = nc % C_total
    # n  = nc // C_total  (not needed explicitly)

    # ── BN parameters for this channel (scalar loads) ──────────────────────
    mean_c   = tl.load(mean_ptr   + c).to(tl.float32)
    var_c    = tl.load(var_ptr    + c).to(tl.float32)
    weight_c = tl.load(weight_ptr + c).to(tl.float32)
    bias_c   = tl.load(bias_ptr   + c).to(tl.float32)

    inv_std = weight_c * tl.rsqrt(var_c + 0.001)
    shift   = bias_c - inv_std * mean_c   # precomputed per-channel offset

    # ── Spatial offsets ─────────────────────────────────────────────────────
    hw_start = hw_blk * BLOCK_SIZE
    hw_offs  = hw_start + tl.arange(0, BLOCK_SIZE)
    mask     = hw_offs < HW

    # ── Select input tensor based on channel ────────────────────────────────
    is_in_a = c < C_A
    # Safe index for tensor_b: clamp c_b to 0 when loading from tensor_a
    c_b = tl.where(is_in_a, 0, c - C_A)

    n_offset  = nc // C_total   # recover n from nc (integer floor)
    a_base    = n_offset * (C_A * HW) + c      * HW
    b_base    = n_offset * (C_B * HW) + c_b    * HW

    val_a = tl.load(tensor_a_ptr + a_base + hw_offs,
                    mask=mask & is_in_a,  other=0.0).to(tl.float32)
    val_b = tl.load(tensor_b_ptr + b_base + hw_offs,
                    mask=mask & ~is_in_a, other=0.0).to(tl.float32)
    val = tl.where(is_in_a, val_a, val_b)

    # ── BN + ReLU ────────────────────────────────────────────────────────────
    y = inv_std * val + shift
    z = tl.maximum(y, 0.0)

    tl.store(out_ptr + nc * HW + hw_offs, z, mask=mask)


# ── Python wrapper (marked @torch.fx.wrap so FX never traces inside) ────────

@torch.fx.wrap
def fused_cat_bn_relu(tensor_a, tensor_b, bn_mean, bn_var, bn_weight, bn_bias):
    """
    Fused: cat([tensor_a, tensor_b], dim=1) → BN(inference) → ReLU

    tensor_a : [N, C_A, H, W]
    tensor_b : [N, C_B, H, W]   (already at the same spatial size as tensor_a)
    Returns  : [N, C_A+C_B, H, W]
    """
    N, C_A, H, W = tensor_a.shape
    C_B     = tensor_b.shape[1]
    C_total = C_A + C_B
    HW      = H * W

    out = torch.empty((N, C_total, H, W),
                      dtype=tensor_a.dtype, device=tensor_a.device)

    grid = lambda meta: (N * C_total, triton.cdiv(HW, meta['BLOCK_SIZE']))

    _fused_cat_bn_relu_kernel[grid](
        tensor_a, tensor_b,
        bn_mean, bn_var, bn_weight, bn_bias,
        out,
        N, C_A, C_B, HW, C_total,
    )
    return out


# ── Pattern / replacement API ────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Matches: cat([in_4, in_5], 1)
             → batch_norm(cat_out, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
             → relu(bn_out, inplace=False)

    Variable mapping (both model variants hit the same structural pattern):
      start73 / start7 : in_0=mean, in_1=var, in_2=bias, in_3=weight,
                         in_4=direct_tensor, in_5=post_pool_tensor
      start1           : in_0=mean, in_1=var, in_2=bias, in_3=weight,
                         in_4=direct_tensor, in_5=post_pool_tensor
                         (framework fills the slots appropriately)
    """
    tmp_6 = torch.cat([in_4, in_5], 1)
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # fused_cat_bn_relu(tensor_a, tensor_b, mean, var, weight, bias)
    return (in_4, in_5, in_0, in_1, in_3, in_2)


def replacement_func():
    return fused_cat_bn_relu