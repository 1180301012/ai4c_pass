"""
Shared Triton kernels and dispatch function for Gemma KV optimizations.
Imported by both FuseRoPEKeyExpandForGemma.py and FuseExpandReshapeValue.py
so that both pass files share the SAME replacement_func object.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Fused RoPE for key_states
#   BLOCK_SIZE=256 ensures the rotate_half gather accesses the SAME
#   256-element block, keeping all loads coalesced in L1.
# ---------------------------------------------------------------------------
@triton.jit
def _kv_rope_kernel(
    key_ptr, cos_ptr, sin_ptr, out_ptr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    key     = tl.load(key_ptr + offsets, mask=mask, other=0.0)
    cos_val = tl.load(cos_ptr + offsets, mask=mask, other=0.0)
    sin_val = tl.load(sin_ptr + offsets, mask=mask, other=0.0)

    # rotate_half: d < HALF_DIM → -key[d+HALF_DIM],  d >= HALF_DIM → key[d-HALF_DIM]
    d = offsets % DIM
    seq_base = offsets - d
    rotate_offsets = seq_base + (d + HALF_DIM) % DIM
    rotate_key = tl.load(key_ptr + rotate_offsets, mask=mask, other=0.0)
    rotate_key = tl.where(d < HALF_DIM, -rotate_key, rotate_key)

    result = key * cos_val + rotate_key * sin_val
    tl.store(out_ptr + offsets, result, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2: Broadcast-expand [1,1,3,256] → [1,8,3,256]
# ---------------------------------------------------------------------------
@triton.jit
def _kv_expand_kernel(
    src_ptr, dst_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    val = tl.load(src_ptr + offsets, mask=mask, other=0.0)

    # broadcast to 8 heads: dst[h, s, d] = src[0, 0, s, d], h*N offset
    tl.store(dst_ptr + 0 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 1 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 2 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 3 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 4 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 5 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 6 * N + offsets, val, mask=mask)
    tl.store(dst_ptr + 7 * N + offsets, val, mask=mask)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper (same function object shared across all passes)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def shared_dispatch(a0, a1, a2, route):
    """
    Route-based dispatch:
      route == "rope"   → RoPE kernel (a0=key, a1=cos, a2=sin)
      route == "expand" → expand kernel (a0=x to broadcast, a1/a2 unused)
    """
    if route == "rope":
        N = 768          # 1*1*3*256
        DIM = 256
        HALF_DIM = 128
        BLOCK_SIZE = 256
        out = torch.empty_like(a0)
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _kv_rope_kernel[grid](
            a0, a1, a2, out,
            N=N, DIM=DIM, HALF_DIM=HALF_DIM, BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
    elif route == "expand":
        N = 768          # 1*1*3*256
        SEQ_LEN = 3
        DIM = 256
        BLOCK_SIZE = 256
        out = torch.empty((1, 8, SEQ_LEN, DIM), dtype=a0.dtype, device=a0.device)
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _kv_expand_kernel[grid](a0, out, N=N, BLOCK_SIZE=BLOCK_SIZE)
        return out