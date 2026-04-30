import torch
import triton
import triton.language as tl


def pattern(x, y, z):
    s = torch.sigmoid(x)
    r = y * s
    out = r + z
    return out


def replacement_args(x, y, z):
    return (x, y, z)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def sigmoid_mul_add_kernel(
    x_ptr, y_ptr, z_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-x))
    result = y * sig + z

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_sigmoid_mul_add(x, y, z):
    out = torch.empty_like(y)
    n_elements = y.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    sigmoid_mul_add_kernel[grid](x, y, z, out, n_elements)
    return out


def replacement_func():
    return fused_sigmoid_mul_add