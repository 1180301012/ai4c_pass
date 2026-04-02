import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Diagnostic: simplest possible pattern matching just "a + b"
# If this matches, subgraph matching works with operator.add level ops.
# ---------------------------------------------------------------------------
def pattern(x, y):
    return x + y


def replacement_args(x, y):
    return (x, y)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def triton_add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N
    x   = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y   = tl.load(y_ptr)  # load scalar (in_0 has shape [1])
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_add(x, y):
    N   = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    triton_add_kernel[grid](x, y, out, N)
    return out


def replacement_func():
    return triton_add