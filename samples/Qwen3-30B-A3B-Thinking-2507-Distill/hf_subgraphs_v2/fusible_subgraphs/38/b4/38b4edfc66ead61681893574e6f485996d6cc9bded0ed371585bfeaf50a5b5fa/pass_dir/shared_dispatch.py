import torch
import triton
import triton.language as tl


# ── Single parameterised Triton kernel ─────────────────────────────────────────
# Covers any N <= BLOCK_N.
#   causal_ptr    : [1,1,N,N] float32  – causal-attention mask, values ∈ {0, −FLT_MAX}
#   attn_mask_ptr : [1,1,N,N] int64   – raw attention mask (0 or 1), broadcast from [1,N]
#   out_ptr       : [1,1,N,N] float32  – −∞ where causal forbids OR attn_mask==0, else 0
#
# Logic: out = −∞  iff  (causal mask forbids access) OR (attn_mask[0,col] == 0)
#          = causal  otherwise
#
# Intuition for the combined mask:
#   attn_mask[0, col] == 0  ↔  original attention_mask[0, col] == 1
#                             (real token at column col)
#   So attn_mask==0 means "do NOT mask this column", and non-zero means "mask it".
#   Combined: −∞ when causal forbids OR attn says to mask.
#
# Comparison trick for float32 causal: use int64 comparison on attn_mask instead
# (avoids float literal precision issues entirely).
@triton.jit
def _attn_causal_kernel(
    causal_ptr,
    attn_mask_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    valid = cols < N

    causal_val = tl.load(causal_ptr + row * N + cols, mask=valid, other=0.0)

    # Load the single relevant attention-mask value for this row/col.
    # For the broadcast tensor [1,1,N,N] with underlying [1,N],
    # element [0,0,row,col] lives at byte-offset (row*N + col) * 8 from the base ptr.
    row_val = tl.load(attn_mask_ptr + row * N + cols, mask=valid, other=1)
    # row_val ∈ {0, 1} (int64).
    # 0 = original attn_mask was 1 → this column is a real token → NOT padding-masked
    # ≠ 0 = original attn_mask was 0 → padding → mask this column

    # −FLT_MAX → causal forbids → output −FLT_MAX
    # 0.0       → causal allows  → output 0
    # Safe float32 comparison: 0.0 > −FLT_MAX is True; −FLT_MAX > −FLT_MAX is False.
    is_valid = causal_val > -3.4028234663852886e+38
    causal_out = tl.where(is_valid, causal_val, -3.4028234663852886e+38)

    # attn_mask==0 → real token (no padding mask) → keep causal_out
    # attn_mask≠0 → padding → output −FLT_MAX
    out = tl.where(row_val == 0, causal_out, -3.4028234663852886e+38)

    tl.store(out_ptr + row * N + cols, out, mask=valid)


# ── Shared dispatch wrapper ────────────────────────────────────────────────────
@torch.fx.wrap
def _attn_causal_mask_dispatch(causal_mask, attn_mask, route):
    # N from tensor shape (handles both N=9 and N=13 graphs)
    N = causal_mask.shape[-1]
    BLOCK_N = 16          # next power-of-2 >= max(N); covers N=9 and N=13
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=causal_mask.device)
    _attn_causal_kernel[(N,)](
        causal_mask, attn_mask, out, N, BLOCK_N=BLOCK_N,
    )
    return out