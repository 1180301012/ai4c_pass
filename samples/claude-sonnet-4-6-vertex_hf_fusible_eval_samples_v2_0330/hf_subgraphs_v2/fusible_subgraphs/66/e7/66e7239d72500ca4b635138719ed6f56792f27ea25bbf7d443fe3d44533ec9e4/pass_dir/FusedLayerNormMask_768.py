import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused kernel: mask-broadcast (int64→float32 expand) + elementwise multiply
#
# Replaces the sequence AFTER layer_norm:
#   tmp_5 = in_0.unsqueeze(-1)          # view, no kernel
#   tmp_6 = tmp_5.expand_as(tmp_4)       # view, no kernel
#   tmp_7 = tmp_6.float()                # materialize broadcast → float32
#   tmp_8 = tmp_4 * tmp_7               # element-wise multiply
#
# Layer_norm runs untouched as PyTorch's native CUDA kernel (fast).
# This kernel only handles the simpler mask+multiply part.
#
# Design:
#   TILE_H = 256, H=768 = 3×256 → N_TILES=3, total programs = 48
#   (48 programs on 56 SMs → ~86% SM utilisation, vs 16 for the full fusion)
#   No reduction needed → very low per-program compute cost.
# ---------------------------------------------------------------------------
@triton.jit
def mask_broadcast_mul_kernel(
    ln_ptr,          # [B*S, H] float16/bfloat16 – layer_norm output (tmp_4)
    mask_ptr,        # [B*S]   int64 – attention mask (in_0 flattened)
    out_mask_ptr,    # [B*S, H] float32 – tmp_7 (broadcast mask)
    out_mul_ptr,     # [B*S, H] float32 – tmp_8 (masked output)
    H, N_TILES,
    TILE_H: tl.constexpr,
):
    pid      = tl.program_id(0)
    row_idx  = pid // N_TILES
    tile_idx = pid  % N_TILES

    # Load scalar mask: int64 → float32
    mask_val = tl.load(mask_ptr + row_idx).to(tl.float32)

    # Column range for this tile
    col_start = tile_idx * TILE_H
    row_start = row_idx  * H
    offsets   = col_start + tl.arange(0, TILE_H)
    valid     = offsets < H

    # Load layer_norm output, upcast to float32
    x     = tl.load(ln_ptr + row_start + offsets, mask=valid, other=0.0)
    x_f32 = x.to(tl.float32)

    # Store tmp_7: broadcast mask as float32
    mask_expanded = tl.zeros((TILE_H,), dtype=tl.float32) + mask_val
    tl.store(out_mask_ptr + row_start + offsets, mask_expanded, mask=valid)

    # Store tmp_8: masked multiply (float16 * float32 → float32 in PyTorch)
    tl.store(out_mul_ptr  + row_start + offsets, x_f32 * mask_val,  mask=valid)


@torch.fx.wrap
def mask_broadcast_mul_launch(in_0, in_4):
    """
    Fuses:
        tmp_5 = in_0.unsqueeze(-1)
        tmp_6 = tmp_5.expand_as(in_4)   # in_4 = tmp_4 (layer_norm output)
        tmp_7 = tmp_6.float()
        tmp_8 = in_4 * tmp_7
    Returns (tmp_7, tmp_8).
    Layer_norm itself is NOT replaced – it runs as native PyTorch.
    """
    B, S, H = in_4.shape
    rows = B * S

    TILE_H  = 256
    N_TILES = (H + TILE_H - 1) // TILE_H   # 3 for H=768
    total   = rows * N_TILES                 # 48

    out_mask = torch.empty((B, S, H), dtype=torch.float32, device=in_4.device)
    out_mul  = torch.empty((B, S, H), dtype=torch.float32, device=in_4.device)

    mask_broadcast_mul_kernel[(total,)](
        in_4,
        in_0.reshape(-1),
        out_mask,
        out_mul,
        H, N_TILES,
        TILE_H=TILE_H,
        num_warps=8,
    )

    return (out_mask, out_mul)


def mask_broadcast_mul(in_0, in_4):
    """
    Non-wrapped: FX traces this to produce 2 getitem nodes,
    matching the pattern's 2 returning nodes.
    """
    result = mask_broadcast_mul_launch(in_0, in_4)
    tmp_7  = result[0]
    tmp_8  = result[1]
    return (tmp_7, tmp_8)


# ─── Pattern / replacement interface ────────────────────────────────────────
#
# Pattern matches ONLY (unsqueeze → expand_as → float → multiply).
# Layer_norm is intentionally LEFT OUTSIDE the pattern so that
# PyTorch's highly optimised native kernel handles it unchanged.

def pattern(in_0, in_4):
    """
    in_0 : attention mask  [B, S]        int64
    in_4 : layer_norm output [B, S, H]   float16/bfloat16  (pattern input = tmp_4)
    """
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(in_4)
    tmp_7 = tmp_6.float()
    tmp_8 = in_4 * tmp_7
    return (tmp_7, tmp_8)


def replacement_args(in_0, in_4):
    return (in_0, in_4)


def replacement_func():
    return mask_broadcast_mul