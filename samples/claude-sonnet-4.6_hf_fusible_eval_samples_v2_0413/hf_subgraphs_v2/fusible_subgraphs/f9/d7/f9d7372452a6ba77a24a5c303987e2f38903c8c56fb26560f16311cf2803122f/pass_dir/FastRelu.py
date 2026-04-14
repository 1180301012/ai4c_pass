import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match any relu with inplace=True
# This reliably matches the target's relu(tmp_0, inplace=True) node.
# ---------------------------------------------------------------------------
def pattern(x):
    return torch.nn.functional.relu(x, inplace=True)


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton fused relu kernel – minimises kernel-launch overhead and uses
# aggressive autotuning for the small [1,128,16,12] tensor size.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 512},  num_warps=2),
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=4),
        triton.Config({'BLOCK': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fast_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid    = tl.program_id(0)
    offs   = pid * BLOCK + tl.arange(0, BLOCK)
    mask   = offs < n_elements
    x      = tl.load(x_ptr + offs, mask=mask, other=0.0)
    result = tl.maximum(x, 0.0)
    tl.store(out_ptr + offs, result, mask=mask)


@torch.fx.wrap
def fast_relu(x):
    n   = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    _fast_relu_kernel[grid](x, out, n)
    return out


def replacement_func():
    return fast_relu