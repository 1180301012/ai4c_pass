import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the full computation in model.py
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: fused weighted-sum / normalise
#
#   in_0 : [B, SEQ_LEN, H]  int64
#   in_1 : [B, SEQ_LEN, H]  fp16/bf16  (promoted to fp32 inside kernel)
#   out  : [B, H]            fp32       (same as eager output dtype)
#
# For each (b, j):
#   num   = sum_i( fp32(in_1[b,i,j]) * fp32(in_0[b,i,j]) )
#   denom = sum_i( fp32(in_0[b,i,j]) )
#   out[b,j] = num / max(denom, 1e-9)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_weighted_avg_kernel(
    in0_ptr,              # int64  [B, SEQ_LEN, H]
    in1_ptr,              # fp16/bf16 [B, SEQ_LEN, H]
    out_ptr,              # fp32   [B, H]
    SEQ_LEN: tl.constexpr,   # always 10 for this graph
    H:       tl.constexpr,   # always 1024 for this graph
    BLOCK_H: tl.constexpr,   # tile width along H dimension
):
    b     = tl.program_id(0)   # batch index
    h_blk = tl.program_id(1)   # H-tile index

    h_off = h_blk * BLOCK_H + tl.arange(0, BLOCK_H)
    mask  = h_off < H

    # fp32 running sums
    num_s = tl.zeros([BLOCK_H], dtype=tl.float32)
    denom = tl.zeros([BLOCK_H], dtype=tl.float32)

    row0 = b * SEQ_LEN * H   # base offset into in_0 for this batch
    row1 = b * SEQ_LEN * H   # base offset into in_1 for this batch
    ro   = b * H             # base offset into out   for this batch

    # Unrolled loop over the short SEQ_LEN=10 axis
    for i in range(SEQ_LEN):
        off = i * H + h_off

        # in_0: int64 → fp32 (safe, in_0 values are 0 or 1)
        v0 = tl.load(in0_ptr + row0 + off, mask=mask, other=0).to(tl.float32)

        # in_1: fp16/bf16 → fp32
        v1 = tl.load(in1_ptr + row1 + off, mask=mask, other=0.0).to(tl.float32)

        num_s += v1 * v0
        denom += v0

    # Normalise (clamp denominator to avoid division by zero)
    denom_c = tl.maximum(denom, 1e-9)
    result  = num_s / denom_c

    # Store as fp32 (matches eager output dtype)
    tl.store(out_ptr + ro + h_off, result, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_weighted_avg(in_0, in_1):
    B       = in_0.shape[0]    # 1
    SEQ_LEN = in_0.shape[1]    # 10
    H       = in_0.shape[2]    # 1024

    # Allocate fp32 output — dtype must match eager (all arithmetic is fp32)
    out = torch.empty((B, H), dtype=torch.float32, device=in_0.device)

    # TILE_H: cover all H=1024 in one tile (BLOCK_H must be power-of-2 >= H)
    TILE_H  = 1024
    N_H     = (H + TILE_H - 1) // TILE_H   # = 1

    _fused_weighted_avg_kernel[(B, N_H)](
        in_0, in_1, out,
        SEQ_LEN=SEQ_LEN,
        H=H,
        BLOCK_H=TILE_H,
        num_warps=16,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func: must return the function object, not call it
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_weighted_avg