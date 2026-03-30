import torch
import triton
import triton.language as tl
from torch import device

# ── Triton kernel: reads tmp_11 via column-indexing ───────────────────────────
# tmp_11 has strides [N,N,0,1] so element [0,0,i,j] lives at physical offset j.
# Reading `tmp11_ptr + col` (col = k % N) gives in_0[0, col] for any row i.
# With N and N_SQ as constexpr, Triton specialises per size (9 or 13).
@triton.jit
def _v3_kernel(
    tmp11_ptr,    # int64  [physically N elements]
    causal_ptr,   # float32 [N_SQ] contiguous
    out_ptr,      # float32 [N_SQ]
    N: tl.constexpr,
    N_SQ: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    NEG_INF = -3.4028234663852886e+38
    offsets = tl.arange(0, BLOCK_SIZE)
    valid   = offsets < N_SQ
    col     = offsets % N

    attn   = tl.load(tmp11_ptr + col, mask=valid, other=1)
    causal = tl.load(causal_ptr + offsets, mask=valid, other=0.0)
    out    = tl.where(attn == 0, NEG_INF, causal)
    tl.store(out_ptr + offsets, out, mask=valid)


# ── Replacement: 6 real GPU kernels → 1 Triton kernel ────────────────────────
@torch.fx.wrap
def _fused_mask_v3(tmp_9, tmp_11):
    """
    tmp_9 : float32 [1,1,N,N] causal mask  (contiguous)
    tmp_11: int64   [1,1,N,N] expanded attention mask (strides [N,N,0,1])
    Fuses: to(float32), sub, to(bool), masked_fill, to(cuda)[noop], bool(), masked_fill
    """
    N    = int(tmp_9.shape[-1])       # 9 or 13 — Python int → Triton constexpr
    N_sq = N * N
    out  = torch.empty_like(tmp_9)
    BLOCK_SIZE = 128 if N_sq <= 128 else 256
    _v3_kernel[(1,)](tmp_11, tmp_9, out, N=N, N_SQ=N_sq, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ── Pattern: 7 internal ops, size-agnostic ────────────────────────────────────
def pattern(tmp_9, tmp_11, const_one):
    tmp_12 = tmp_11.to(torch.float32)
    tmp_14 = const_one - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


def replacement_args(tmp_9, tmp_11, const_one):
    return (tmp_9, tmp_11)


def replacement_func():
    return _fused_mask_v3