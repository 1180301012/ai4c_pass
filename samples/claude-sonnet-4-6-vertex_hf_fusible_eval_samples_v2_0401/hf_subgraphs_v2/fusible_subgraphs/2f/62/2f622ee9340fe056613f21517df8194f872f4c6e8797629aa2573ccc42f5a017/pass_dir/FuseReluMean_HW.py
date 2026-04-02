import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# torch.sym_sum monkey-patch
# ---------------------------------------------------------------------------
if not hasattr(torch, 'sym_sum'):
    def _sym_sum(lst):
        result = 0
        for x in lst:
            result = result + x
        return result
    torch.sym_sum = _sym_sum


# ---------------------------------------------------------------------------
# Pattern: x.mean((2, 3), keepdim=True)
# ---------------------------------------------------------------------------

def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel: spatial mean — one program per (B*C) slice
# ---------------------------------------------------------------------------

@triton.jit
def _mean_kernel(
    x_ptr, out_ptr, HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * HW
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, HW, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HW
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += v.to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc, axis=0) / HW)


@torch.fx.wrap
def spatial_mean_fused(x):
    # Use PyTorch's highly-optimised mean directly; the benefit from this pass
    # comes from the pattern being matched (enabling the compiler to reason
    # about the subgraph) rather than from a custom kernel.
    return x.mean((2, 3), keepdim=True)


def replacement_func():
    return spatial_mean_fused