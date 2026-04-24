import torch
import triton
import triton.language as tl
from torch import device

# ---------------------------------------------------------------------------
# Pattern: torch.arange(1, device=cuda:0)
# ---------------------------------------------------------------------------
def pattern():
    tmp_0 = torch.arange(1, device=device(type='cuda', index=0))
    return tmp_0


def replacement_args():
    return ()


# ---------------------------------------------------------------------------
# Triton kernel: fill output with [0, 1, ..., n-1]
# For n=1 this is simply [0]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1}),
        triton.Config({'BLOCK_SIZE': 2}),
        triton.Config({'BLOCK_SIZE': 4}),
        triton.Config({'BLOCK_SIZE': 8}),
        triton.Config({'BLOCK_SIZE': 16}),
    ],
    key=['n_elements'],
)
@triton.jit
def arange_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, offsets.to(tl.int64), mask=mask)


@torch.fx.wrap
def triton_arange_1():
    n_elements = 1
    out = torch.empty(n_elements, dtype=torch.int64, device='cuda')
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    arange_kernel[grid](
        out_ptr=out,
        n_elements=n_elements,
    )
    return out


def replacement_func():
    return triton_arange_1