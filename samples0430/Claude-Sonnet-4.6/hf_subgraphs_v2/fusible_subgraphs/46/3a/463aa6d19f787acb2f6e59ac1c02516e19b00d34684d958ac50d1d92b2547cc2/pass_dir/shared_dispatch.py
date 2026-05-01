"""Shared Triton kernels and dispatch wrapper imported by all pass files."""
import torch
import triton
import triton.language as tl


@triton.jit
def _zero_neginf_rows_ip(data_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    """In-place: zero rows where ALL values == NEG_INF. Valid rows untouched."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    NEG_INF = -3.4028234663852886e+38
    vals = tl.load(data_ptr + row * N + cols, mask=col_mask, other=NEG_INF)
    is_valid = vals != NEG_INF
    n_valid = tl.sum(is_valid.to(tl.int32), 0)
    if n_valid <= 0:
        tl.store(data_ptr + row * N + cols,
                 tl.zeros([BLOCK_N], dtype=tl.float32), mask=col_mask)


@triton.jit
def _build_causal_mask(out_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    """Build causal mask: NEG_INF where col > row, 0 elsewhere."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    NEG_INF = -3.4028234663852886e+38
    out_val = tl.where(cols > row,
                       tl.full([BLOCK_N], NEG_INF, dtype=tl.float32),
                       tl.zeros([BLOCK_N], dtype=tl.float32))
    tl.store(out_ptr + row * N + cols, out_val, mask=col_mask)


@torch.fx.wrap
def _dispatch(arg0, arg1, route):
    """
    Shared replacement function for ALL passes (same object → no limit issues).
    Routes:
      "TAIL"     - in-place zero all-NEG_INF rows in arg0
      "HEAD_N21" - build causal mask N=21
      "HEAD_N10" - build causal mask N=10
      "HEAD_N13" - build causal mask N=13
    """
    if route == "TAIL":
        N = arg0.shape[-1]
        BLOCK_N = 32 if N > 16 else 16
        _zero_neginf_rows_ip[(N,)](arg0, N=N, BLOCK_N=BLOCK_N)
        return arg0
    elif route == "HEAD_N21":
        out = torch.empty((1, 1, 21, 21), dtype=torch.float32, device=arg0.device)
        _build_causal_mask[(21,)](out, N=21, BLOCK_N=32)
        return out
    elif route == "HEAD_N10":
        out = torch.empty((1, 1, 10, 10), dtype=torch.float32, device=arg0.device)
        _build_causal_mask[(10,)](out, N=10, BLOCK_N=16)
        return out
    elif route == "HEAD_N13":
        out = torch.empty((1, 1, 13, 13), dtype=torch.float32, device=arg0.device)
        _build_causal_mask[(13,)](out, N=13, BLOCK_N=16)
        return out
    else:
        return arg0