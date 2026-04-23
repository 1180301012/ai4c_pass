import torch
import triton
import triton.language as tl


def pattern(x, y):
    return x + y


def replacement_args(x, y):
    return (x, y)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


@torch.fx.wrap
def triton_add(x, y):
    out = torch.empty_like(x)
    n_elements = x.numel()
    _add_kernel[(triton.cdiv(n_elements, 1024),)](
        x,
        y,
        out,
        n_elements,
    )
    return out


def replacement_func():
    return triton_add