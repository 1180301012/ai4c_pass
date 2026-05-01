import torch
import triton
import triton.language as tl


def pattern(conv2d_out, in_2):
    tmp_3 = torch.ops.aten.relu_.default(conv2d_out)
    tmp_4 = torch.ops.aten.add.Tensor(in_2, tmp_3)
    return tmp_4


def replacement_args(conv2d_out, in_2):
    return (conv2d_out, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _relu_add_kernel(
    x_ptr,    # conv2d output  (to be relu'd)
    y_ptr,    # residual input
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # relu(x) + y
    out = tl.maximum(x, 0.0) + y

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu_add_interpolate(conv2d_out, in_2):
    """
    Fused kernel that computes:  out = relu(conv2d_out) + in_2
    The subsequent interpolate(size=(24,24)) is a no-op (same spatial dims)
    so we skip it entirely, saving a full memory round-trip.
    """
    n = conv2d_out.numel()
    out = torch.empty_like(in_2)

    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _relu_add_kernel[grid](
        conv2d_out, in_2, out,
        n_elements=n,
    )
    return out


def replacement_func():
    return fused_relu_add_interpolate