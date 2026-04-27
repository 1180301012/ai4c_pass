import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – mirrors model.py exactly (method calls, keyword dim=1, etc.)
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel – fuses all 8 ops into one kernel pass.
#
# Performance strategy:
#   num_warps=1  →  only 32 threads per block, NO __syncthreads barriers,
#   only intra-warp shuffles for the prefix scan.  This eliminates the
#   cross-warp synchronisation overhead that dominates for small sequences.
#   BLOCK_S = ceil_pow2(S): each of the 32 threads handles BLOCK_S/32 elems.
# ---------------------------------------------------------------------------
@triton.jit
def _position_ids_kernel(
    input_ptr,
    output_ptr,
    B, S,
    stride_b, stride_s,
    BLOCK_S: tl.constexpr,
):
    """
    result[b,s] = cumsum(x[b] != 1)[s] * (x[b,s] != 1) + 1
    Input: int64 [B, S].   Output: int64 [B, S].
    One block per row; BLOCK_S = next-power-of-2(S).
    Launched with num_warps=1 to use only warp shuffles (no __syncthreads).
    """
    batch_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_S)
    mask_valid = offsets < S

    # Load int64 tokens; out-of-bounds slots → value 1 (padding → m=0)
    x = tl.load(
        input_ptr + batch_id * stride_b + offsets * stride_s,
        mask=mask_valid, other=1
    )

    # int32 attention mask: 1 = real token, 0 = padding
    m = (x != 1).to(tl.int32)

    # Inclusive prefix-sum of mask along the sequence
    cs = tl.cumsum(m, axis=0)

    # Position IDs: zero-out padding, add 1
    result = (cs * m + 1).to(tl.int64)

    tl.store(
        output_ptr + batch_id * stride_b + offsets * stride_s,
        result,
        mask=mask_valid
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
# Cache (S → (BLOCK_S, num_warps)) to avoid re-creating large Python int
# objects (>256) on every call, reducing Python GC pressure.
_BLOCK_S_CACHE = {}


@torch.fx.wrap
def position_ids_fused(in_0):
    B, S = in_0.shape

    # Look up cached config to avoid recomputing large ints each call
    config = _BLOCK_S_CACHE.get(S)
    if config is None:
        BLOCK_S = max(32, 1 << (max(S, 1) - 1).bit_length())
        # BLOCK_S ≤ 512: num_warps=1  → 0 barriers (warp shuffles only)
        # BLOCK_S = 1024: num_warps=2 → 1 barrier  (16 elems/thread, safe)
        num_warps = 1 if BLOCK_S <= 512 else 2
        config = (BLOCK_S, num_warps)
        _BLOCK_S_CACHE[S] = config
    BLOCK_S, num_warps = config

    # Avoid tuple creation: torch.empty(B, S, ...) vs torch.empty((B, S), ...)
    out = torch.empty(B, S, dtype=torch.int64, device=in_0.device)

    # Pass grid as tuple (Triton requires tuple for grid, not plain int)
    _position_ids_kernel[(B,)](
        in_0, out,
        B, S,
        in_0.stride(0), in_0.stride(1),
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


def replacement_func():
    return position_ids_fused