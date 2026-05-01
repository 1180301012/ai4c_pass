import torch
import triton
import triton.language as tl


# ─── Pattern ────────────────────────────────────────────────────────────────

def pattern(x, scale):
    """Match: tensor / scalar  followed by  transpose(-1, -2) — fuse both into one Triton kernel."""
    tmp_0 = x / scale
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1


def replacement_args(x, scale):
    return (x, scale)


# ─── 2-D tile kernel: fully-coalesced reads AND writes via tl.trans() ────────
# Input  shape: [..., S, D] → Output shape: [..., D, S]
# Each CTA handles one (bh, s_tile):
#   Load  [BLOCK_S, BLOCK_D] tile  →  contiguous reads  (D-stride = 1)
#   tl.trans → [BLOCK_D, BLOCK_S] in registers (zero-cost register reorder)
#   Store [BLOCK_D, BLOCK_S] tile  →  contiguous writes (S-stride = 1)
# When BLOCK_D = D and BLOCK_S = S: both reads and writes cover a single
# contiguous memory block → fully coalesced with no cache-line waste.

@triton.heuristics({
    'BLOCK_S': lambda args: (
        16  if args['S'] <= 16  else
        64  if args['S'] <= 64  else
        128 if args['S'] <= 128 else 256
    ),
    'BLOCK_D': lambda args: 8 if args['D'] <= 8 else 64,
})
@triton.jit
def _div_trans_2d_kernel(
    in_ptr, out_ptr,
    S, D,
    scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s  = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)

    s_mask = s_offs < S
    d_mask = d_offs < D
    mask_sd = s_mask[:, None] & d_mask[None, :]    # [BLOCK_S, BLOCK_D]

    # Load [BLOCK_S, BLOCK_D] tile – contiguous block in input memory
    in_base    = pid_bh * S * D
    in_offsets = in_base + s_offs[:, None] * D + d_offs[None, :]
    tile = tl.load(in_ptr + in_offsets, mask=mask_sd, other=0.0)
    tile = tile / scale

    # Transpose in registers (zero-cost): [BLOCK_S, BLOCK_D] → [BLOCK_D, BLOCK_S]
    tile_T = tl.trans(tile)

    # Store [BLOCK_D, BLOCK_S] tile – contiguous block in output memory
    out_base    = pid_bh * D * S
    out_offsets = out_base + d_offs[:, None] * S + s_offs[None, :]
    mask_ds     = d_mask[:, None] & s_mask[None, :]
    tl.store(out_ptr + out_offsets, tile_T, mask=mask_ds)


@torch.fx.wrap
def _fused_div_transpose(x, scale):
    S   = x.shape[-2]
    D   = x.shape[-1]
    BH  = x.numel() // (S * D)
    out = torch.empty(x.shape[:-2] + (D, S), dtype=x.dtype, device=x.device)
    # BLOCK_S/BLOCK_D match actual dims for zero waste; static grid avoids meta dict
    BLOCK_S = 16 if S <= 16 else (64 if S <= 64 else (128 if S <= 128 else 256))
    BLOCK_D = 8  if D <= 8  else 64
    _div_trans_2d_kernel[(BH, (S + BLOCK_S - 1) // BLOCK_S)](x, out, S, D, scale)
    return out


# ─── Entry point for the pass framework ──────────────────────────────────────

def replacement_func():
    return _fused_div_transpose