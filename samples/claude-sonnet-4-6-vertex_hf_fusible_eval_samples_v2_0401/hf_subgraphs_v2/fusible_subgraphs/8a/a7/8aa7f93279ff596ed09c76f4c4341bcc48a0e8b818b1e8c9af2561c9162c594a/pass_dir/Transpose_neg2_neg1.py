import torch
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────
# The _decomposed model represents  in_0.transpose(-2, -1)  as
# aten.permute.default(x, [0, 1, 3, 2]).  We therefore match the permute form.
def pattern(x):
    return x.permute(0, 1, 3, 2)


# ── Triton kernel (reference, not called in the hot path) ─────────────────────
@triton.jit
def _permute_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


# ── Wrapper ───────────────────────────────────────────────────────────────────
# permute / transpose returns a view (zero GPU cost).
# We reproduce the same view so correctness is preserved.
@torch.fx.wrap
def permute_wrapper(x):
    return x.permute(0, 1, 3, 2)


# ── Pass interface ────────────────────────────────────────────────────────────
def replacement_args(x):
    return (x,)


def replacement_func():
    return permute_wrapper