import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – matches the elementwise multiply (a * b)
# In the model this is:  tmp_4 = in_2 * tmp_3
# ---------------------------------------------------------------------------

def pattern(a, b):
    return a * b


def replacement_args(a, b):
    return (a, b)


# ---------------------------------------------------------------------------
# Triton kernel – fast elementwise multiply in native dtype (no f32 upcasting)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_sigmoid_interp_mul(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a * b, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper – must be decorated with @torch.fx.wrap
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_sigmoid_interp_mul(a, b):
    """Elementwise multiply via Triton, native dtype, fixed BLOCK_SIZE."""
    out = torch.empty_like(a)
    n   = a.numel()
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _fused_sigmoid_interp_mul[grid](a, b, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# replacement_func – zero-arg, returns callable
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_sigmoid_interp_mul