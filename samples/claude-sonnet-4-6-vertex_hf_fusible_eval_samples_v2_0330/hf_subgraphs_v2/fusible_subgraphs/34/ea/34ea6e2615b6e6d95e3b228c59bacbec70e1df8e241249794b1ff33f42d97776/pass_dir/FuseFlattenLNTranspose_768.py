import torch
import triton
import triton.language as tl


# ── Pattern: exactly 1 returning node ────────────────────────────────────────
# layer_norm output (ln) is referenced by the two downstream transposes,
# both outside the matched subgraph → 1 unique returning node.
# match.returning_nodes=1, copied_returning_nodes=1 → assertion passes.
def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-05)


def replacement_args(x, weight, bias):
    return (x, weight, bias)


# ── Triton kernel ─────────────────────────────────────────────────────────────
# Input x = tmp_7: shape [B, N, 768], strides [B_s, 1, 256] (NCHW-derived view).
# Access x[b, n, c] = x_ptr + n*stride_n + c*stride_c.
# For N=256 concurrent programs, channel-c reads are at stride 256 apart but
# consecutive programs read consecutive addresses → coalesced across programs.
# Single-pass variance (E[x²]−E[x]²) — numerically safe when |mean|≪std.
# All arithmetic in float32 for bfloat16 stability.

@triton.jit
def _ln_bf16_768(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    stride_n,           # x.stride(1) = 1  for NCHW-derived view
    stride_c,           # x.stride(2) = N  for NCHW-derived view
    N,                  # x.shape[1] = H*W = 256
    C,                  # x.shape[2] = 768
    eps,
    BLOCK_C: tl.constexpr,
):
    n    = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    # Single load (held in registers for both reductions)
    x = tl.load(x_ptr + n * stride_n + cols * stride_c,
                 mask=mask, other=0.0).to(tl.float32)

    # Single-pass mean and variance
    xm  = tl.where(mask, x,     0.0)
    x2m = tl.where(mask, x * x, 0.0)
    mean = tl.sum(xm,  axis=0) / C
    var  = tl.sum(x2m, axis=0) / C - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    w   = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b_  = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = tl.where(mask, (x - mean) * rstd * w + b_, 0.0)

    tl.store(y_ptr + n * C + cols, out.to(tl.bfloat16), mask=mask)


# ── Replacement wrapper ───────────────────────────────────────────────────────
@torch.fx.wrap
def opt_ln_768(x, weight, bias):
    """
    x      : [B, N, 768]  bfloat16 (tmp_7, non-contiguous NCHW-derived view)
    weight : [768]
    bias   : [768]
    Returns layer_norm output [B, N, 768] contiguous bfloat16.
    """
    B, N, C = x.shape
    sn = x.stride(1)
    sc = x.stride(2)
    y  = torch.empty(B, N, C, dtype=x.dtype, device=x.device)
    _ln_bf16_768[(B * N,)](x, weight, bias, y, sn, sc, N, C, 1e-5,
                           BLOCK_C=1024, num_warps=4)
    return y


def replacement_func():
    return opt_ln_768