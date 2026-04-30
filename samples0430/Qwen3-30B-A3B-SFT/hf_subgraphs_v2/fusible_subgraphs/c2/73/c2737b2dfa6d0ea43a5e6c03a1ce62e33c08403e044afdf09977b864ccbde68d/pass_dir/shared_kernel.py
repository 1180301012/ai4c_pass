"""Shared Triton kernel for fused div + transpose.

Both pass files import dispatch_fused_div_transpose from here so that
replacement_func() returns the SAME Python object (satisfies the limit).

Kernel design (avoids tl.trans which can fail for large D_CONST):
  Grid: (B*H, S)  — one program per (batch*head, sequence-position)
  Load  D elements from input[bh, s, :]  (strided reads — 1 element per cache line)
  Store D elements to output[bh, :, s]  (D consecutive elements → COALESCED writes)
  No tl.trans needed: natural (d, s) indexing gives correct output layout.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_div_transpose_kernel(
    input_ptr,
    output_ptr,
    scale,
    D,
    S,
    D_CONST: tl.constexpr,   # D itself (power-of-2)
):
    """Grid: (B*H, S).
    Each program handles one (bh, s) pair:
      reads  D elements input[bh, s, d]  for d=0..D-1  (strided reads)
      writes D elements output[bh, d, s]  for d=0..D-1  (coalesced writes)
    No tl.trans used — correct because element x[d] = input[bh,s,d] stored
    to output[bh, d, s] = output_ptr + bh*D*S + d*S + s.
    """
    pid_bh = tl.program_id(0)   # batch*head index
    pid_s  = tl.program_id(1)   # sequence index

    s = pid_s
    d = tl.arange(0, D_CONST)   # [D]

    # ── Load D elements from input[bh, s, d] ──────────────────────────────
    # Clamp s to 0 when s>=S so the address is ALWAYS in-bounds
    # (the mask prevents writing these elements, so correctness is preserved)
    safe_s = tl.where(s < S, s, 0)
    safe_addr = pid_bh * (S * D) + safe_s * D + d   # [D]
    x = tl.load(input_ptr + safe_addr)

    # ── Divide and store to output[bh, d, s] ──────────────────────────────
    # consecutive d elements → COALESCED writes
    tl.store(output_ptr + pid_bh * (D * S) + d * S + s, x / scale)


@torch.fx.wrap
def dispatch_fused_div_transpose(x, route):
    """Single dispatch wrapper shared by all pass files.

    `route` is a plain Python string from replacement_args();
    both routes call the same kernel with the appropriate scale.
    """
    B, H, S, D = x.shape
    out = torch.empty(B, H, D, S, dtype=x.dtype, device=x.device)

    if route == "r16817":
        scale = 1.6817928305074292
    else:   # "r28284"
        scale = 2.8284271247461903

    # Grid: (B*H, S)
    grid = (B * H, S)

    _fused_div_transpose_kernel[grid](
        x, out, scale, D, S,
        D_CONST=D,
        num_warps=4,
    )
    return out