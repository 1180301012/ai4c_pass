import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_flatten_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def relu_flatten(in_0):
    B = in_0.shape[0]
    n_elements = in_0.numel()
    out = torch.empty((B, n_elements // B), dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    relu_flatten_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
    )
    return out


def replacement_func():
    return relu_flatten