import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Fused kernel: BN (inference) + Add + ReLU + Spatial Mean
#
# Grid: (NC, num_tiles)
#   NC     = B * C  – one program per (batch, channel) pair
#   num_tiles = ceil(HW / BLOCK_HW)
#
# Each program handles one tile of HW elements.
# A partial sum is accumulated; the first tile (pid==0) stores the final mean.
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128},  num_warps=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_bn_add_relu_mean_kernel(
    x_ptr,       # [B, C, H, W]  input for BN  (in_4)
    y_ptr,       # [B, C, H, W]  input for add  (in_5)
    mean_ptr,    # [B, C, 1, 1]  output mean    (tmp_7)
    rm_ptr,      # [C]           running_mean   (in_0)
    rv_ptr,      # [C]           running_var    (in_1)
    w_ptr,       # [C]           weight         (in_3)
    b_ptr,       # [C]           bias           (in_2)
    out_ptr,     # [B, C, H, W]  output         (tmp_6)
    C, HW,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)   # which (batch, channel)
    pid_t  = tl.program_id(1)   # which tile along HW

    c = pid_nc % C

    # ── Load per-channel BN parameters (scale + bias) ──────────────────────
    rm  = tl.load(rm_ptr + c).to(tl.float32)
    rv  = tl.load(rv_ptr + c).to(tl.float32)
    wt  = tl.load(w_ptr  + c).to(tl.float32)
    bt  = tl.load(b_ptr  + c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(rv + eps)
    scale   = wt * inv_std                        # precompute fused scale

    # ── Offsets for this tile ──────────────────────────────────────────────
    start    = pid_t * BLOCK_HW
    offsets  = start + tl.arange(0, BLOCK_HW)
    mask     = offsets < HW
    base     = pid_nc * HW

    # ── Load, apply BN, add, ReLU ──────────────────────────────────────────
    xv = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    yv = tl.load(y_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)

    val = yv + (xv - rm) * scale + bt
    val = tl.maximum(val, 0.0)                    # ReLU

    tl.store(out_ptr + base + offsets, val.to(xv.dtype), mask=mask)

    # ── Partial sum contribution to the mean ───────────────────────────────
    acc = tl.sum(tl.where(mask, val, 0.0), axis=0)

    # Only the first tile in the HW dimension contributes to the global mean
    if pid_t == 0:
        total_sum  = tl.sum(acc, axis=0)
        total_elements = tl.cast(HW, tl.float32)
        mean_val    = (total_sum / total_elements).to(out_ptr.dtype.element_ty)
        out_mean_idx = pid_nc                          # mean is B*C contiguous
        tl.store(mean_ptr + out_mean_idx, mean_val)


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper  (called by the replacement; must be @torch.fx.wrap)
# ──────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_bn_add_relu_mean(rm, rv, bt, wt, x, y):
    """
    rm : running_mean   [C]
    rv : running_var    [C]
    bt : bias           [C]
    wt : weight         [C]
    x  : input for BN   [B, C, H, W]
    y  : residual input [B, C, H, W]
    returns (out [B,C,H,W], mean_out [B,C,1,1])
    """
    B, C, H, W = x.shape
    HW        = H * W
    NC        = B * C
    num_tiles = triton.cdiv(HW, 4096)   # upper bound; autotune will override

    out       = torch.empty_like(x)
    mean_out  = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)

    # Grid is a lambda so autotune can adjust BLOCK_HW → num_tiles correctly
    def grid(meta):
        blocks_per_nc = triton.cdiv(HW, meta['BLOCK_HW'])
        return (NC, blocks_per_nc)

    _fused_bn_add_relu_mean_kernel[grid](
        x, y, mean_out,
        rm, rv, wt, bt,
        out,
        C, HW,
        eps=1e-5,
    )

    return out, mean_out


# ──────────────────────────────────────────────────────────────────────────────
# Pass API
# ──────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Mirrors model.py exactly, including argument order and values."""
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return (tmp_6, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # (rm, rv, bias, weight, x, y)  – matches the fused_bn_add_relu_mean signature
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_bn_add_relu_mean