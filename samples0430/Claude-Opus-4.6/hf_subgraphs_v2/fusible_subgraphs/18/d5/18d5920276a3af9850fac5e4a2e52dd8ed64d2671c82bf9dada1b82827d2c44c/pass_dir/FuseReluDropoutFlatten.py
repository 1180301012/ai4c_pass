import torch
import triton
import triton.language as tl


# Pattern matching function - mirrors the model.py operations exactly
def pattern(x):
    tmp_0 = torch.nn.functional.relu(x, inplace=False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
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
    # ReLU
    zero = tl.zeros_like(x)
    out = tl.maximum(x, zero)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu_flatten(x):
    # Input shape is [B, C, 1, 1], output shape is [B, C]
    batch_size = x.shape[0]
    channels = x.shape[1]
    n_elements = batch_size * channels

    # Allocate output with flattened shape
    out = torch.empty(batch_size, channels, device=x.device, dtype=x.dtype)

    # Launch kernel
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    relu_flatten_kernel[grid](
        x,
        out,
        n_elements,
    )

    return out


def replacement_func():
    return fused_relu_flatten