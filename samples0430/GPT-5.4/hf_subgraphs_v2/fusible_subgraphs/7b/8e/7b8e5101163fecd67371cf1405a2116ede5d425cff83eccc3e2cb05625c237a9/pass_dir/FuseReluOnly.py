import torch
import triton
import triton.language as tl


def pattern(in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    return tmp_2


def replacement_args(in_2):
    return (in_2,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    zero = tl.zeros((BLOCK_SIZE,), dtype=x.dtype)
    y = tl.maximum(x, zero)
    tl.store(out_ptr + offs, y, mask=mask)


@torch.fx.wrap
def triton_relu(in_2):
    out = torch.empty_like(in_2)
    n_elements = in_2.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](in_2, out, n_elements)
    return out


def replacement_func():
    return triton_relu