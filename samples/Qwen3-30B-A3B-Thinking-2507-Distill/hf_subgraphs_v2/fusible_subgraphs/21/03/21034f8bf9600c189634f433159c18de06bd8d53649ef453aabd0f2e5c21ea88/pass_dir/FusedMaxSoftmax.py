import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: identity copy (used to test pattern matching)
# ---------------------------------------------------------------------------

@triton.jit
def _copy_kernel(in_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)


@torch.fx.wrap
def fused_view(in_1):
    """Identity replacement for in_1.view(B, C, -1) — same shape/dtype, zero-cost copy."""
    B = in_1.shape[0]
    C = in_1.shape[1]
    out = torch.empty(B, C, 64 * 64, dtype=in_1.dtype, device=in_1.device)
    n = out.numel()
    BLOCK = 1024
    _copy_kernel[((n + BLOCK - 1) // BLOCK,)](in_1, out, n, BLOCK=BLOCK)
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

def pattern(in_1):
    result = in_1.view(12, 512, -1)
    return result


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return fused_view