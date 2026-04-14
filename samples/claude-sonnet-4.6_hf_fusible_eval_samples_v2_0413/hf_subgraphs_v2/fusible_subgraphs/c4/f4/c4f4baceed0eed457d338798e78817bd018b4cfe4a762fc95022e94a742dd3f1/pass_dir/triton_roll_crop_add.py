"""
Triton kernel for fused roll + crop + residual_add (single output).

Fuses: view(-1,H,H,C) + roll(3,3) + slice + contiguous + view(1,N,C) + add
→ single pass over data, no intermediate allocations.

The pattern does NOT include the first in_3.contiguous() call.
The pattern's `in_3` input corresponds to `tmp_2` (already contiguous) in the FX graph.
The pattern returns a SINGLE tensor (tmp_8), avoiding multi-output tuple issues.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_roll_crop_add_kernel(
    in3_ptr,    # contiguous, effective [B, H, H, C]
    in2_ptr,    # [B, CROP*CROP, C]
    out_ptr,    # [B, CROP*CROP, C]
    H:       tl.constexpr,
    CROP:    tl.constexpr,
    C:       tl.constexpr,
    SHIFT:   tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid   = tl.program_id(0)
    N_out = CROP * CROP
    b     = pid // N_out
    token = pid % N_out
    out_r = token // CROP
    out_c = token % CROP

    # Inverse of roll: source indices in the original (pre-roll) tensor
    src_r = (out_r - SHIFT + H) % H
    src_c = (out_c - SHIFT + H) % H

    in3_base = (b * H * H + src_r * H + src_c) * C
    tok_base = (b * N_out + token) * C

    c_range = tl.arange(0, BLOCK_C)
    mask    = c_range < C

    x = tl.load(in3_ptr + in3_base + c_range, mask=mask, other=0.0)
    y = tl.load(in2_ptr + tok_base  + c_range, mask=mask, other=0.0)

    # Same dtype arithmetic as in_2 + tmp_7 in the original model
    tl.store(out_ptr + tok_base + c_range, x + y, mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Shared dispatch wrapper — all three pass files return THIS function so the
# pass manager sees only ONE unique replacement_func.
# Returns a SINGLE tensor (the residual sum), no tuple.
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_roll_crop_add_dispatch(in_2, in_3, route):
    B   = in_2.shape[0]
    out = torch.empty_like(in_2)

    if route == "96":
        total = B * 128 * 128
        fused_roll_crop_add_kernel[(total,)](
            in_3, in_2, out,
            H=133, CROP=128, C=96, SHIFT=3, BLOCK_C=128,
        )
    elif route == "192":
        total = B * 64 * 64
        fused_roll_crop_add_kernel[(total,)](
            in_3, in_2, out,
            H=70, CROP=64, C=192, SHIFT=3, BLOCK_C=256,
        )
    else:  # "384"
        total = B * 32 * 32
        fused_roll_crop_add_kernel[(total,)](
            in_3, in_2, out,
            H=35, CROP=32, C=384, SHIFT=3, BLOCK_C=512,
        )

    return out