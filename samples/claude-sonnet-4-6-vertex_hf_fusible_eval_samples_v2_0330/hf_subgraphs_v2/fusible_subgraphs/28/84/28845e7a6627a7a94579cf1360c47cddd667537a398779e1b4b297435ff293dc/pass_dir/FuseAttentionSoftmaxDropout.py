"""
DIAGNOSTIC PASS: Testing whether the framework supports subgraph matching.

Uses the REFERENCE EXAMPLE from the instructions: pattern(a, b) -> a + b
This should match the 'in_1 + in_0' node in all 5 target graphs.

If this matches: subgraph matching IS supported → fix the specific ops.
If this FAILS:   framework requires FULL GRAPH matching → different strategy.
"""
import torch
import triton
import triton.language as tl


@torch.fx.wrap
def fused_add(a, b):
    """Identity replacement for a+b — used only as diagnostic."""
    return a + b


@torch.fx.wrap
def fused_softmax_dropout(x):
    """
    Drop-in replacement for softmax(dim=-1) + dropout(training=False).
    Works for any 3D input shape [N, S, S] and any floating-point dtype.
    """
    assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D"
    N, S1, S2 = x.shape
    rows = N * S1
    S    = S2
    BLOCK_S = 16 if S <= 16 else (32 if S <= 32 else 64)
    out = torch.empty_like(x)
    _softmax_kernel[(rows,)](x, out, S, BLOCK_S=BLOCK_S)
    return out


@triton.jit
def _softmax_kernel(
    x_ptr, out_ptr,
    S,
    BLOCK_S: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    x    = tl.load(x_ptr + pid * S + offs, mask=mask, other=float('-inf'))
    x32  = x.to(tl.float32)
    xm   = tl.max(x32, axis=0)
    xe   = tl.exp(x32 - xm)
    xe   = tl.where(mask, xe, 0.0)
    xs   = tl.sum(xe, axis=0)
    tl.store(out_ptr + pid * S + offs, (xe / xs).to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# DIAGNOSTIC: minimal add pattern (reference example from instructions)
# ---------------------------------------------------------------------------

def pattern(a, b):
    """Matches a+b subgraph — the simplest possible test."""
    return a + b


def replacement_args(a, b):
    return (a, b)


def replacement_func():
    return fused_add