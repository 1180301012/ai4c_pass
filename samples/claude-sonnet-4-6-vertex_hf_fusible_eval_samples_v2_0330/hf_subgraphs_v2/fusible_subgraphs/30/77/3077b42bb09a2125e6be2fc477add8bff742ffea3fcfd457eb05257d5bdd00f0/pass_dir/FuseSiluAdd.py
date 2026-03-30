import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0 + in_0
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def silu_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # SiLU: y * sigmoid(y)
    silu_y = y * tl.sigmoid(y)

    out = silu_y + x
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def silu_add(in_0, in_1):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    silu_add_kernel[grid](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
    )
    return out


def replacement_func():
    return silu_add