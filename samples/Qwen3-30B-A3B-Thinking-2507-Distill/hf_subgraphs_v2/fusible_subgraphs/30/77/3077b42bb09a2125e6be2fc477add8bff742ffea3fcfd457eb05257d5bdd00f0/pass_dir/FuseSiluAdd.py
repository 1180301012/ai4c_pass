import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: add-only — matches operator.add(in_0=silu_output, in_1=in_0_model)
# This confirms operator.add is matchable and gives correct output.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    return in_0 + in_1


def replacement_args(in_0, in_1):
    # in_0 = silu output (first add operand), in_1 = in_0 model (second add)
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: plain element-wise add
# in_0 = silu output, in_1 = in_0 model  → computes out = in_0 + in_1
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_add(in_0, in_1):
    N = in_0.numel()
    out = torch.empty_like(in_0)
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _add_kernel[grid](in_0, in_1, out, N)
    return out


def replacement_func():
    return triton_add