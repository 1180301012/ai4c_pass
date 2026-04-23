import torch
import triton
import triton.language as tl


_DIVISOR = 11.313708498984761
_INV_DIVISOR = 0.08838834764831843


def pattern(in_0):
    tmp_0 = in_0 / 11.313708498984761
    tmp_1 = torch.nn.functional.relu(tmp_0)
    tmp_2 = torch.square(tmp_1)
    return (tmp_2,)


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit

def _fused_div_relu_square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)
    y = x * _INV_DIVISOR
    y = tl.maximum(y, 0.0)
    y = y * y
    tl.store(out_ptr + offsets, y, mask=mask)


@torch.fx.wrap
def fused_div_relu_square_sqrt128(in_0):
    x = in_0
    if not x.is_contiguous():
        x = x.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _fused_div_relu_square_kernel[grid](
        x,
        out,
        n_elements,
    )
    return (out,)


def replacement_func():
    return fused_div_relu_square_sqrt128