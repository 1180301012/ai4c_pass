import torch
import triton
import triton.language as tl
from torch import device
import operator


def pattern(x, pos_embed):
    """Minimal test: match just the add; x=inputs_embeds, pos_embed=position weights."""
    added = operator.add(x, pos_embed)
    return added


def replacement_args(x, pos_embed):
    return (x, pos_embed)


@triton.jit
def _add_kernel(x_ptr, pos_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    p = tl.load(pos_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + p, mask=mask)


@torch.fx.wrap
def fused_add_layernorm(x, pos_embed):
    """Triton-backed add (diagnostic: tests if operator.add matches in graph)."""
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _add_kernel[grid](x, pos_embed, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return fused_add_layernorm