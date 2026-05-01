"""
Shared Triton kernel for full-graph fusion:
  conv2d([B,64,20,20], w[1,64,1,1], bias[1]) → view(B,1,400)
  → cat([in3[B,1,6400], in4[B,1,1600], view], dim=2) → [B,1,8400]
  → sigmoid → sub(0.25) → mul(pi)

Single-kernel design: ONE launch replaces all 6 FX nodes, eliminating
cuDNN overhead and 5 extra kernel-launch gaps.

tl.static_range(IN_CHAN) fully unrolls the 64-channel loop so the compiler
can fold k*400 to compile-time constants, reducing index arithmetic.
"""
import torch
import triton
import triton.language as tl

# ── compile-time constants ─────────────────────────────────────────────────
_BLOCK_SIZE = 256   # elements per thread-block
_N_BLOCKS   = 33    # ceil(8400 / 256)
_IN_CHAN    = 64    # conv input channels


@triton.jit
def _full_fused_kernel(
    bias_ptr,      # [1]
    weight_ptr,    # [64]  — weight[0, :, 0, 0]
    conv_in_ptr,   # [B, 64, 400]  — in_2 viewed as flat
    in3_ptr,       # [B, 6400]     — in_3 viewed as flat
    in4_ptr,       # [B, 1600]     — in_4 viewed as flat
    out_ptr,       # [B, 8400]     — output
    B,
    IN_CHAN:    tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    bid   = tl.program_id(1)

    pos = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    in3_mask  = (pos < 6400)
    in4_mask  = (pos >= 6400) & (pos < 8000)
    conv_mask = (pos >= 8000) & (pos < 8400)
    valid     = (pos < 8400)

    # ── in3 segment ──────────────────────────────────────────────────────
    idx3 = tl.where(in3_mask, batch * 6400 + pos, 0)
    x3   = tl.load(in3_ptr + idx3, mask=in3_mask, other=0.0)

    # ── in4 segment ──────────────────────────────────────────────────────
    idx4 = tl.where(in4_mask, batch * 1600 + (pos - 6400), 0)
    x4   = tl.load(in4_ptr + idx4, mask=in4_mask, other=0.0)

    # ── conv segment (1×1 conv) ───────────────────────────────────────────
    sp = tl.where(conv_mask, pos - 8000, 0)   # spatial pos in [0, 400)

    bias = tl.load(bias_ptr)
    acc  = tl.where(conv_mask, bias.to(tl.float32), 0.0)

    # Fully-unrolled loop: k is a compile-time constant each iteration,
    # so `k * 400` and `weight_ptr + k` become constant offsets.
    # The `if bid >= CONV_BID` skips the entire loop for the 31 non-conv
    # blocks per batch, which is the vast majority of all blocks.
    # This runtime branch is warp-uniform → no divergence, just a fast skip.
    CONV_BID: tl.constexpr = 8000 // BLOCK_SIZE   # = 31 for BLOCK_SIZE=256
    if bid >= CONV_BID:
        for k in tl.static_range(IN_CHAN):
            w_k   = tl.load(weight_ptr + k)
            i_idx = tl.where(conv_mask,
                             batch * IN_CHAN * 400 + k * 400 + sp, 0)
            i_k   = tl.load(conv_in_ptr + i_idx, mask=conv_mask, other=0.0)
            acc   = acc + i_k.to(tl.float32) * w_k.to(tl.float32)

    # ── combine ───────────────────────────────────────────────────────────
    val = x3.to(tl.float32) + x4.to(tl.float32) + acc

    # ── sigmoid → sub → mul ───────────────────────────────────────────────
    val = tl.sigmoid(val)
    val = (val - 0.25) * 3.141592653589793

    tl.store(out_ptr + batch * 8400 + pos, val.to(x3.dtype), mask=valid)


@torch.fx.wrap
def full_fused_conv_cat_sigmoid(bias, weight, conv_in, in3, in4):
    B   = in3.shape[0]
    out = torch.empty((B, 1, 8400), dtype=in3.dtype, device=in3.device)

    _full_fused_kernel[(B, _N_BLOCKS)](
        bias, weight, conv_in, in3, in4, out, B,
        IN_CHAN=_IN_CHAN,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
    return out