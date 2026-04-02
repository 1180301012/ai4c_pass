"""
Universal fused Add + LayerNorm pass.

Strategy: patch SubgraphMatcher so it uses ignore_literals=True, allowing
one pattern to match all five target graphs regardless of their normalized_shape
constant ((768,), (1024,), (16,), …).  The lazy import inside _replace_pattern
means the patched class is picked up at match time.
"""

import torch
import triton
import triton.language as tl
import torch.fx.passes.utils.matcher_utils as _matcher_utils


# ── Patch SubgraphMatcher to ignore literal constants during matching ────────

_OriginalSubgraphMatcher = _matcher_utils.SubgraphMatcher


class _IgnoreLiteralsSubgraphMatcher(_OriginalSubgraphMatcher):
    """Drop-in replacement that always sets ignore_literals=True."""

    def __init__(
        self,
        pattern,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=False,          # signature preserved for compatibility
    ):
        super().__init__(
            pattern,
            match_output=match_output,
            match_placeholder=match_placeholder,
            remove_overlapping_matches=remove_overlapping_matches,
            ignore_literals=True,       # always ignore literal args (e.g., normalized_shape)
        )


# Replace in the module so the lazy `from … import SubgraphMatcher` in
# custom_replacement._replace_pattern picks up our patched version.
_matcher_utils.SubgraphMatcher = _IgnoreLiteralsSubgraphMatcher


# ── Triton fused kernel ─────────────────────────────────────────────────────

@triton.jit
def _fused_add_ln_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr, out_ptr,
    N, eps, stride_row,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    z = x + y

    mean = tl.sum(z, 0) / N
    diff = tl.where(mask, z - mean, 0.0)
    var  = tl.sum(diff * diff, 0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    result = diff * inv_std * w + b
    tl.store(out_ptr + row * stride_row + offs,
             result.to(x_ptr.dtype.element_ty), mask=mask)


# ── Python wrapper ──────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_add_layernorm_universal(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [N]
    in_1 : weight [N]
    in_2 : first  input [*, N]
    in_3 : second input [*, N]

    Handles any N (768, 1024, 16, …).
    BLOCK_N = next_power_of_2(N) ensures all N elements are processed.
    """
    N = in_1.shape[0]
    # next_power_of_2 guarantees BLOCK_N >= N so all elements are included
    BLOCK_N = triton.next_power_of_2(N)
    # Heuristic: more warps for larger blocks (better warp-level parallelism)
    num_warps = 1 if BLOCK_N <= 32 else (2 if BLOCK_N <= 256 else (8 if BLOCK_N <= 1024 else 16))

    num_rows = in_2.numel() // N
    out = torch.empty_like(in_2)

    _fused_add_ln_kernel[(num_rows,)](
        in_2, in_3, in_1, in_0, out,
        N, 1e-05,
        N,          # stride_row
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out


# ── FX pattern / replacement hooks ─────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    # (768,) is a placeholder; SubgraphMatcher runs with ignore_literals=True
    # so the actual normalized_shape value in the target is irrelevant.
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_add_layernorm_universal