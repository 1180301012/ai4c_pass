import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    # Add-only pattern to test if operator.add matches the model's add.
    # If model uses operator.add, this should match.
    return in_0 + in_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def _silu_add_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in1_ptr + offsets, mask=mask, other=0.0)

    # Plain element-wise add (no silu - silu already computed by model)
    out = x + y

    tl.store(out_ptr + offsets, out.to(x.dtype), mask=mask)


@torch.fx.wrap
def silu_add(x, y):
    # For add-only match: x = silu(in_1), y = in_0
    # We compute x + y directly (just the add, no silu since silu was already computed)
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    _silu_add_kernel[grid](x, y, out, N)
    return out


def replacement_func():
    return silu_add