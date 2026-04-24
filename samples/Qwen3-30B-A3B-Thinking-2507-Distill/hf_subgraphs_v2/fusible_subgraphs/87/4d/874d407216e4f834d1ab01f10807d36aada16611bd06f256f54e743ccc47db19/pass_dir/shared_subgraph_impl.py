"""
Shared Triton kernel + Python wrapper for broadcast-multiply optimization.

Both FuseViewMulExpand_1000_16.py and FuseViewMulExpand_128_128.py import
_dispatch_full_subgraph from THIS file so that replacement_func() in both
pass files returns the EXACT SAME Python object — satisfying the
replacement_func_limit constraint.
"""
import torch
import triton
import triton.language as tl


# ── Triton kernel ─────────────────────────────────────────────────────────────

@triton.jit
def _bcast_mul_kernel(
    in1_ptr,
    in2_ptr,
    out_ptr,
    N,
    M,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_M)
    mask = cols < M

    w = tl.load(in1_ptr + row)                          # scalar weight per row
    x = tl.load(in2_ptr + row * M + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * M + cols, w * x, mask=mask)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run_bcast_mul(in_1, in_2):
    """Launch broadcast-multiply kernel; returns [N, M] output tensor."""
    N = in_2.shape[0]
    M = in_2.shape[1]
    out = torch.empty((N, M), dtype=in_2.dtype, device=in_2.device)
    if M <= 16:
        _bcast_mul_kernel[(N,)](in_1, in_2, out, N, M, BLOCK_M=16)
    else:
        _bcast_mul_kernel[(N,)](in_1, in_2, out, N, M, BLOCK_M=128)
    return out


# ── Opaque dispatch wrapper (shared across both pass files) ───────────────────

@torch.fx.wrap
def _dispatch_full_subgraph(in_1, in_2, route="1000_16"):
    """
    Single opaque entry-point for the broadcast-mul replacement.
    'route' is provided by replacement_args; default keeps old passes working.
    Both pass files return this same object → satisfies replacement_func_limit.
    Dispatches to _run_bcast_mul which selects BLOCK_M based on M at runtime.
    """
    if route == "1000_16":
        return _run_bcast_mul(in_1, in_2)
    elif route == "128_128":
        return _run_bcast_mul(in_1, in_2)
    # fallback — should never be reached
    return _run_bcast_mul(in_1, in_2)