import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – matches in_0.view(1,2,1,8,8).expand(1,2,64,8,8)
# Returns a single tensor so the return-node count stays at 1.
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_2 = in_0.view(1, 2, 1, 8, 8)
    tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel: broadcast-copy [B,C,H,W] → [B,C,E,H,W]
#
#   out[b, c, e, h, w] = in[b, c, h, w]   (e dimension broadcast-repeated)
#
# All shape constants are baked in as tl.constexpr for B=1,C=2,E=64,H=8,W=8.
# TOTAL=8192, BLOCK=1024 → 8 programs; each processes 1024 output elements.
# The 128-element input (256 B fp16) fits in L1 → reads effectively free.
# Writes to the 16 KB output are fully coalesced.
# ---------------------------------------------------------------------------
@triton.jit
def broadcast_expand_kernel(
    in_ptr,
    out_ptr,
    BLOCK: tl.constexpr,   # 1024
    TOTAL: tl.constexpr,   # 8192  = B*C*E*H*W
    HW:    tl.constexpr,   # 64    = H*W
    E:     tl.constexpr,   # 64
    H:     tl.constexpr,   # 8
    W:     tl.constexpr,   # 8
    C:     tl.constexpr,   # 2
):
    pid      = tl.program_id(0)
    out_offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask     = out_offs < TOTAL

    # Decompose output flat index → (c, e, h, w) for B=1
    w_idx = out_offs % W
    tmp   = out_offs // W
    h_idx = tmp % H
    tmp   = tmp // H
    tmp   = tmp // E          # skip the 'e' dimension
    c_idx = tmp               # == c  (since B=1)

    # Input flat index: ignores 'e'
    in_offs = c_idx * HW + h_idx * W + w_idx

    vals = tl.load(in_ptr + in_offs, mask=mask)
    tl.store(out_ptr + out_offs, vals, mask=mask)


# ---------------------------------------------------------------------------
# Replacement wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def materialized_expand(in_0):
    B, C, E, H, W = 1, 2, 64, 8, 8
    TOTAL  = B * C * E * H * W   # 8192
    HW     = H * W               # 64
    BLOCK  = 1024
    n_prog = TOTAL // BLOCK       # 8  (exact division)

    out = torch.empty((B, C, E, H, W), dtype=in_0.dtype, device=in_0.device)

    broadcast_expand_kernel[(n_prog,)](
        in_0.reshape(-1),
        out.reshape(-1),
        BLOCK=BLOCK,
        TOTAL=TOTAL,
        HW=HW,
        E=E, H=H, W=W, C=C,
        num_warps=4,
    )
    return out


def replacement_func():
    return materialized_expand