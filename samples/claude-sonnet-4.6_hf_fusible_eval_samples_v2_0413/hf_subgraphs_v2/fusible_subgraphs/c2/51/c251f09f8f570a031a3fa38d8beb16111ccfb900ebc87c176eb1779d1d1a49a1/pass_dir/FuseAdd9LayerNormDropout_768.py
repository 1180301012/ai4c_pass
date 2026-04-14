import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Pattern: 8 sequential additions (9 embedding inputs) + LayerNorm + Dropout
# Fusing all these into one kernel:
#   - Eliminates 7 intermediate tensor writes (especially important when
#     B*S*768*4 > L2 size, e.g. B=128, S=64 → 25 MB tensors exceed 24 MB L2)
#   - LayerNorm included to avoid an extra DRAM round-trip for the sum
# -----------------------------------------------------------------------

def pattern(e0, e1, e2, e3, e4, e5, e6, e7, e8, ln_weight, ln_bias):
    s = e0 + e1
    s = s + e2
    s = s + e3
    s = s + e4
    s = s + e5
    s = s + e6
    s = s + e7
    s = s + e8
    normed = torch.nn.functional.layer_norm(s, (768,), ln_weight, ln_bias, 1e-12)
    out = torch.nn.functional.dropout(normed, 0.1, False, False)
    return out


def replacement_args(e0, e1, e2, e3, e4, e5, e6, e7, e8, ln_weight, ln_bias):
    return (e0, e1, e2, e3, e4, e5, e6, e7, e8, ln_weight, ln_bias)


# -----------------------------------------------------------------------
# Triton kernel: fuse 9-way sum + LayerNorm in one pass.
# 1-D grid: one block per (batch, seq) row.
# e0/e8: shape [B, S, 768] → rB = row * N
# e1..e7: shape [1, S, 768] → rS = (row % seq_len) * N  (broadcast)
# BLOCK_SIZE=1024 covers N=768 with masking for the last 256 slots.
# -----------------------------------------------------------------------

@triton.jit
def _fused_add9_ln_kernel(
    e0_ptr, e1_ptr, e2_ptr, e3_ptr, e4_ptr,
    e5_ptr, e6_ptr, e7_ptr, e8_ptr,
    w_ptr, b_ptr,
    out_ptr,
    seq_len,                   # S (runtime)
    N: tl.constexpr,           # = 768
    BLOCK_SIZE: tl.constexpr,  # = 1024
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    rB = row * N
    rS = (row % seq_len) * N

    # Accumulate 9 embeddings in float32
    x  = tl.load(e0_ptr + rB + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e1_ptr + rS + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e2_ptr + rS + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e3_ptr + rS + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e4_ptr + rS + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e5_ptr + rS + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e6_ptr + rS + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e7_ptr + rS + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e8_ptr + rB + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm: single-pass, sum-of-squares variance
    # Masked positions carry 0.0, so they don't perturb the sums.
    sum_x  = tl.sum(x)
    sum_x2 = tl.sum(x * x)
    mean   = sum_x  / N
    var    = sum_x2 / N - mean * mean
    rstd   = 1.0 / tl.sqrt(var + 1e-12)
    xn     = (x - mean) * rstd

    lw  = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    lb  = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = xn * lw + lb

    tl.store(out_ptr + rB + cols, out, mask=mask)


@torch.fx.wrap
def fused_add9_ln(e0, e1, e2, e3, e4, e5, e6, e7, e8, ln_weight, ln_bias):
    N          = 768
    BLOCK_SIZE = 1024

    B       = e0.shape[0]
    S       = e0.shape[1]
    n_rows  = B * S
    seq_len = S

    out = torch.empty_like(e0)

    _fused_add9_ln_kernel[(n_rows,)](
        e0, e1, e2, e3, e4, e5, e6, e7, e8,
        ln_weight, ln_bias,
        out,
        seq_len,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,    # 256 threads → 100 % warp occupancy on A30
        num_stages=1,
    )

    return out


def replacement_func():
    return fused_add9_ln