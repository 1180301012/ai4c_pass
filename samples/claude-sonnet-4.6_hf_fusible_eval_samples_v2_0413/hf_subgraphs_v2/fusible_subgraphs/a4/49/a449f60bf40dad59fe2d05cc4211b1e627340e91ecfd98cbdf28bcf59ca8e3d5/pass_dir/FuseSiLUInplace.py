import torch
import triton
import triton.language as tl
from pass_dir.pattern_builder import build_pattern_gm


# ---------------------------------------------------------------------------
# Pattern: single-node GraphModule — call_function[silu](in_0,) {inplace:True}
# Single-output avoids multi-output replacement complexity.
# ---------------------------------------------------------------------------
pattern = build_pattern_gm()


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel: element-wise SiLU  x * sigmoid(x)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper: single-input, single-output Triton SiLU
# ---------------------------------------------------------------------------
@torch.fx.wrap
def silu_triton(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_kernel[grid](in_0, out, n_elements)
    return out


# ---------------------------------------------------------------------------
# replacement_func: returns the callable (NOT called here)
# ---------------------------------------------------------------------------
def replacement_func():
    return silu_triton