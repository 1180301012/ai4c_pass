"""
Fused pass: (in_0 / 8.0) + in_2 + in_1  →  single Triton kernel
Inputs:
  in_0 : [2, 12, 7, 7]  (attention_scores)
  in_1 : [2,  1, 1, 7]  (extended_attention_mask, broadcast over dims 1,2)
  in_2 : [2, 12, 7, 7]  (values)
Output: [2, 12, 7, 7]

The target FX graph records `tmp_0 += in_2` as operator.iadd, but
torch.fx.Proxy.__iadd__ is not defined → falls back to operator.add.
We monkey-patch Proxy.__iadd__ so the pattern creates an operator.iadd
node that matches the target, enabling full 3-op fusion.
"""

import operator

import torch
import torch.fx
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Monkey-patch FX Proxy.__iadd__ so that `tmp_0 += in_2` in the pattern
# function creates an operator.iadd node instead of falling back to add.
# ---------------------------------------------------------------------------
def _proxy_iadd(self, other):
    return self.tracer.create_proxy(
        'call_function', operator.iadd, (self, other), {}
    )

if not getattr(torch.fx.Proxy, '_ai4c_iadd_patched', False):
    torch.fx.Proxy.__iadd__ = _proxy_iadd
    torch.fx.Proxy._ai4c_iadd_patched = True


# ---------------------------------------------------------------------------
# Pattern – matches all three ops: div, iadd, add
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_0 = in_0 / 8.0
    tmp_0 += in_2       # → operator.iadd node (thanks to patched __iadd__)
    tmp_2 = tmp_0 + in_1
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel – fixed 1D grid, BLOCK_SIZE=1024, no autotune overhead.
# D3=7 and D1D2D3=588 are tl.constexpr so LLVM uses the fast multiply-shift
# reciprocal path for `% 7` and `// 588`.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_div_iadd_add_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out_ptr,
    n_elements,
    D3: tl.constexpr,      # 7
    D1D2D3: tl.constexpr,  # 588
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements

    x0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(in2_ptr + offsets, mask=mask, other=0.0)

    # Broadcast index for in_1 [2,1,1,7]: constexpr D3, D1D2D3 → fast path
    j  = offsets % D3
    b  = offsets // D1D2D3
    x1 = tl.load(in1_ptr + b * D3 + j, mask=mask, other=0.0)

    tl.store(out_ptr + offsets, x0 * 0.125 + x2 + x1, mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_div_iadd_add(in_0, in_1, in_2):
    n_elements = in_0.numel()                                    # 1176
    out = torch.empty_like(in_0)

    D3     = in_0.shape[-1]                                      # 7
    D1D2D3 = in_0.shape[1] * in_0.shape[2] * in_0.shape[3]      # 588

    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE     # 2

    _fused_div_iadd_add_kernel[(num_blocks,)](
        in_0, in_1, in_2, out,
        n_elements, D3, D1D2D3, BLOCK_SIZE,
    )

    return out


# ---------------------------------------------------------------------------
# Replacement function – returns the callable (do NOT call it)
# ---------------------------------------------------------------------------
def replacement_func():
    return _fused_div_iadd_add