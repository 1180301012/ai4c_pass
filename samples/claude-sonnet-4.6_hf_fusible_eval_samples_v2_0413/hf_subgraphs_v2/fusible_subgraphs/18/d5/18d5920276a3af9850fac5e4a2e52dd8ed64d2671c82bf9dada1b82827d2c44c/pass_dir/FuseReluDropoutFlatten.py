import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = torch.nn.functional.dropout(in_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_flatten_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # ReLU: max(x, 0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def relu_flatten(in_0):
    B = in_0.shape[0]
    n_elements = in_0.numel()
    flat_size = n_elements // B
    # Output has flattened shape [B, C] matching flatten(1, -1) on [B, C, H, W]
    out = torch.empty((B, flat_size), dtype=in_0.dtype, device=in_0.device)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    relu_flatten_kernel[grid](in_0, out, n_elements)
    return out


def replacement_func():
    return relu_flatten