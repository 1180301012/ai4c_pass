import torch
import triton
import triton.language as tl


def pattern(tmp_0):
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1


def replacement_args(tmp_0):
    return (tmp_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_relu_flatten_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    # Just copy - flatten is a reshape/view
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_relu_flatten(tmp_0):
    B = tmp_0.shape[0]
    N = tmp_0.numel()
    out = torch.empty((B, N // B), dtype=tmp_0.dtype, device=tmp_0.device)
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_relu_flatten_kernel[grid](
        in_ptr=tmp_0,
        out_ptr=out,
        n_elements=N,
    )
    return out


def replacement_func():
    return fused_relu_flatten