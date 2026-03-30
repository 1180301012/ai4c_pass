import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: 8 sequential additions of 9 embedding vectors
# ---------------------------------------------------------------------------
# The 8 left-to-right additions in PyTorch spawn 8 separate CUDA kernels.
# Fusing them into one eliminates 7 kernel-launch round-trips.
#
# IMPORTANT: We accumulate in the INPUT's native dtype (no float32 upcast)
# and use the SAME left-to-right order as PyTorch.  This guarantees
# BIT-PERFECT results for float32, float16, and bfloat16.
#
# Broadcasting layout:
#   e1 (word_emb)  : [B, S, D]  – full batch
#   e2..e8         : [1, S, D]  – batch=1, broadcast over B
#   e9 (tok_type)  : [B, S, D]  – full batch
# ---------------------------------------------------------------------------

def pattern(e1, e2, e3, e4, e5, e6, e7, e8, e9):
    s = e1 + e2
    s = s + e3
    s = s + e4
    s = s + e5
    s = s + e6
    s = s + e7
    s = s + e8
    return s + e9


def replacement_args(e1, e2, e3, e4, e5, e6, e7, e8, e9):
    return (e1, e2, e3, e4, e5, e6, e7, e8, e9)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _fused_emb_sum_kernel(
    e1_ptr, e2_ptr, e3_ptr, e4_ptr, e5_ptr,
    e6_ptr, e7_ptr, e8_ptr, e9_ptr,
    out_ptr,
    N,                    # B * S  (total token positions)
    S,                    # sequence length (for broadcast index)
    D:       tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    # Clamp padded lanes to index 0 (always intra-row valid address)
    safe = tl.where(mask, offs, 0)

    full_base  = row * D        # for [B, S, D] tensors
    bcast_base = (row % S) * D  # for [1, S, D] tensors (broadcast)

    # Accumulate in NATIVE dtype – no float32 conversion!
    # This mirrors PyTorch's sequential additions exactly → bit-perfect.
    x  = tl.load(e1_ptr + full_base  + safe)   # word emb        [B,S,D]
    x += tl.load(e2_ptr + bcast_base + safe)   # pos emb         [1,S,D]
    x += tl.load(e3_ptr + bcast_base + safe)   # x1 emb          [1,S,D]
    x += tl.load(e4_ptr + bcast_base + safe)   # y1 emb          [1,S,D]
    x += tl.load(e5_ptr + bcast_base + safe)   # x2 emb          [1,S,D]
    x += tl.load(e6_ptr + bcast_base + safe)   # y2 emb          [1,S,D]
    x += tl.load(e7_ptr + bcast_base + safe)   # h emb           [1,S,D]
    x += tl.load(e8_ptr + bcast_base + safe)   # w emb           [1,S,D]
    x += tl.load(e9_ptr + full_base  + safe)   # token-type emb  [B,S,D]

    # Store only valid elements; padded lanes are masked out (no OOB write)
    tl.store(out_ptr + full_base + safe, x, mask=mask)


# ---------------------------------------------------------------------------
# Python-level wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_emb_sum(e1, e2, e3, e4, e5, e6, e7, e8, e9):
    # e1: [B, S, D];  e2..e8: [1, S, D];  e9: [B, S, D]
    B = e1.shape[0]
    S = e2.shape[-2]
    N = e1.numel() // 768    # = B * S

    out = torch.empty_like(e1)   # same shape [B, S, 768] and same dtype

    _fused_emb_sum_kernel[(N,)](
        e1, e2, e3, e4, e5, e6, e7, e8, e9,
        out,
        N=N, S=S,
        D=768, BLOCK_D=1024,
        num_warps=16,
    )
    return out


# ---------------------------------------------------------------------------
# replacement_func
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_emb_sum