import torch
import triton
import triton.language as tl

# In _decomposed graphs, torch.nn.functional.relu(x, inplace=True)
# is lowered to aten.relu_.default (in-place ATen relu).
def pattern(tmp_0):
    tmp_2 = torch.ops.aten.relu_.default(tmp_0)
    return tmp_2


def replacement_args(tmp_0):
    return (tmp_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def _relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    zeros = tl.zeros_like(x)
    out = tl.where(x > zeros, x, zeros)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_relu_inplace(tmp_0):
    N = tmp_0.numel()
    out = torch.empty_like(tmp_0)
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _relu_kernel[grid](tmp_0, out, n_elements=N)
    return out


def replacement_func():
    return triton_relu_inplace