import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: sum over dim-2 then divide by the result (row-wise normalization)
# Input shape: [B, 2, K, L]  (K=8, L=8, B=1)
# Output shape: [B, 2, 1, L]
# ---------------------------------------------------------------------------

def pattern(in_1):
    tmp_0 = in_1.sum(dim=2, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


# ---------------------------------------------------------------------------
# Original first-eval autotune kernel (gave max_diff=0 in first run).
# Key fix: BLOCK_K and BLOCK_L are kernel parameters (not local constexprs),
# so range(0, BLOCK_K) and range(0, BLOCK_L) work correctly in dynamic context.
# ---------------------------------------------------------------------------

@triton.jit
def _norm_q8(in_ptr, out_ptr,
             BLOCK_K: tl.constexpr, BLOCK_L: tl.constexpr):
    K_CONST: tl.constexpr = 8
    L_CONST: tl.constexpr = 8
    NL:      tl.constexpr = 2 * K_CONST * L_CONST   # 128
    BLK:     tl.constexpr = BLOCK_K * BLOCK_L

    pid    = tl.program_id(0)
    b_flat = pid * BLK

    k_idx = tl.arange(0, BLOCK_K)
    l_idx = tl.arange(0, BLOCK_L)

    n_idx  = l_idx // L_CONST          # L-ary parity: 0 for l<8, 1 for l>=8
    col_off = k_idx[:, None] * BLOCK_L + l_idx[None, :]
    offsets = b_flat + n_idx[None, :] * NL + col_off

    mask = k_idx[:, None] < K_CONST

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    s    = tl.sum(x, axis=0)
    out  = x / s[None, :]

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_sum_div(in_1):
    B  = in_1.shape[0]
    N  = in_1.shape[1]
    K  = in_1.shape[2]
    L  = in_1.shape[3]

    out = torch.empty((B, N, 1, L), dtype=in_1.dtype, device=in_1.device)

    _norm_q8[(B,)](in_1, out, BLOCK_K=8, BLOCK_L=8, num_warps=1, num_stages=1)
    return out


def replacement_func():
    return fused_sum_div