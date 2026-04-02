"""
EliminateDeadPoolerFuseAddLN.py

Matches the COMPLETE pattern:
  add → layer_norm → getitem[:, 0] → linear → tanh   (returns layer_norm output)

The getitem→linear→tanh chain is DEAD CODE (not returned by the model).
This pass replaces all five ops with a single fused Triton add+LN kernel,
eliminating the dead-code GEMV (load 384×384 weight matrix) and tanh.

If this pattern does not match (e.g., DCE was already applied upstream),
the fallback pass FuseAddLayerNorm_384 handles just the add+LN.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: add + layer_norm + getitem[:, 0] + linear + tanh
# Returns only tmp_6 (the layer_norm output) – the rest is dead code.
# ---------------------------------------------------------------------------
def pattern(in_6, in_5, weight, bias, W, b):
    tmp_5   = in_6 + in_5
    tmp_6   = torch.nn.functional.layer_norm(tmp_5, (384,), weight, bias, 1e-12)
    tmp_7   = tmp_6[(slice(None, None, None), 0)]
    linear  = torch.nn.functional.linear(tmp_7, W, b)
    tmp_9   = torch.tanh(linear)
    return tmp_6


def replacement_args(in_6, in_5, weight, bias, W, b):
    # W and b are accepted (pattern inputs) but NOT forwarded to the kernel –
    # the dead-code GEMV + tanh is dropped entirely.
    return (in_6, in_5, weight, bias)


# ---------------------------------------------------------------------------
# Triton kernel: fused element-wise add + layer-norm
# One CTA processes ROWS_PER_PROG consecutive rows, sharing gamma/beta loads.
# ---------------------------------------------------------------------------
@triton.jit
def _add_ln_kernel(
    x_ptr,                       # in_6
    y_ptr,                       # in_5
    weight_ptr,                  # gamma  [N]
    bias_ptr,                    # beta   [N]
    out_ptr,                     # output [num_rows, N]
    num_rows,                    # 578
    N,                           # 384
    eps,                         # 1e-12
    BLOCK_N: tl.constexpr,       # 512  (power-of-2 ≥ N)
    ROWS_PER_PROG: tl.constexpr, # 4
):
    prog_id  = tl.program_id(0)
    base_row = prog_id * ROWS_PER_PROG

    off      = tl.arange(0, BLOCK_N)
    col_mask = off < N

    # Load gamma/beta once, reuse for all rows in this CTA
    weight = tl.load(weight_ptr + off, mask=col_mask, other=1.0).to(tl.float32)
    bias_v = tl.load(bias_ptr   + off, mask=col_mask, other=0.0).to(tl.float32)

    for i in tl.static_range(ROWS_PER_PROG):
        row_idx   = base_row + i
        row_valid = row_idx < num_rows
        eff_mask  = col_mask & row_valid
        row_start = row_idx * N

        x = tl.load(x_ptr + row_start + off, mask=eff_mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + row_start + off, mask=eff_mask, other=0.0).to(tl.float32)
        s = x + y

        mean = tl.sum(s, axis=0) / N
        diff = tl.where(eff_mask, s - mean, 0.0)
        var  = tl.sum(diff * diff, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)

        result = diff * rstd * weight + bias_v
        tl.store(out_ptr + row_start + off,
                 result.to(out_ptr.dtype.element_ty),
                 mask=eff_mask)


_BN  = 512
_RPP = 4
_NW  = 4


@torch.fx.wrap
def _fused_add_ln_no_dead(in_6, in_5, weight, bias):
    x = in_6.contiguous()
    y = in_5.contiguous()
    N        = x.shape[-1]
    num_rows = x.numel() // N
    out      = torch.empty_like(x)
    np       = triton.cdiv(num_rows, _RPP)
    _add_ln_kernel[(np,)](
        x, y, weight, bias, out,
        num_rows=num_rows, N=N, eps=1e-12,
        BLOCK_N=_BN, ROWS_PER_PROG=_RPP, num_warps=_NW,
    )
    return out


def replacement_func():
    return _fused_add_ln_no_dead