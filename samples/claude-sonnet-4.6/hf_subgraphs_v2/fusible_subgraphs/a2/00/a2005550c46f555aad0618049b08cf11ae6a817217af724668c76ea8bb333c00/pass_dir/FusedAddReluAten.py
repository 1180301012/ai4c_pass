import torch
import triton
import triton.language as tl


# ATen-level fallback: add → aten.relu.default

def pattern(x, y):
    tmp = x + y
    return torch.ops.aten.relu.default(tmp)


def replacement_args(x, y):
    return (x, y)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _add_relu_aten_kernel(
    x_ptr, y_ptr, out_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x   = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y   = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = tl.where(x + y > 0.0, x + y, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


@torch.fx.wrap
def triton_add_relu_aten(x, y):
    N   = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _add_relu_aten_kernel[grid](x, y, out, N)
    return out


def replacement_func():
    return triton_add_relu_aten