import torch
from torch import device
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – matches the exact computation from model.py
# torch.arange(1, device=device(type='cuda', index=0))  →  tensor([0])
# No tensor inputs; the entire subgraph is captured as a zero-arg pattern.
# ---------------------------------------------------------------------------
def pattern():
    tmp_0 = torch.arange(1, device=device(type='cuda', index=0))
    return tmp_0


def replacement_args():
    """No tensor arguments – the operation is fully determined by constants."""
    return ()


# ---------------------------------------------------------------------------
# Triton kernel: write 0..N-1 into out_ptr (arange semantics)
# ---------------------------------------------------------------------------
@triton.jit
def arange_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # arange value at position i is simply i
    values = offsets.to(tl.int64)
    tl.store(out_ptr + offsets, values, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper – allocates output and launches kernel
# Must be decorated with @torch.fx.wrap so the FX tracer treats it as a leaf.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_arange_1():
    """Replacement for torch.arange(1, device='cuda').

    torch.arange(1) == tensor([0], dtype=torch.int64).
    We use a Triton kernel to fill the single-element output, avoiding the
    overhead of the generic torch.arange implementation.
    """
    n = 1
    out = torch.empty(n, dtype=torch.int64, device="cuda")

    BLOCK_SIZE = 16  # power-of-2, covers n=1 with one CTA
    grid = (triton.cdiv(n, BLOCK_SIZE),)  # = (1,)

    arange_kernel[grid](
        out_ptr=out,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ---------------------------------------------------------------------------
# replacement_func – must be a zero-arg callable that returns the wrapper fn
# ---------------------------------------------------------------------------
def replacement_func():
    return triton_arange_1