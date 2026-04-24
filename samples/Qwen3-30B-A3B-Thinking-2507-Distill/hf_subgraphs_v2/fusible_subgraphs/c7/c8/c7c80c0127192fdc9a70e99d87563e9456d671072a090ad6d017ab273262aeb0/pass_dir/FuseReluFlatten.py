import torch
import triton
import triton.language as tl


def pattern(in_0):
    # Match only the flatten operation (no kwargs normalization issues).
    # in_0 here is the relu output from the original model node.
    # The replacement applies only the flatten (no relu — it was already applied).
    return in_0.flatten(1, -1)


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def flatten_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_flatten(in_0):
    batch = in_0.shape[0]
    n_elements = in_0.numel()
    out = torch.empty(batch, n_elements // batch, dtype=in_0.dtype, device=in_0.device)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    flatten_kernel[grid](in_0, out, n_elements)
    return out


def replacement_func():
    return fused_flatten