import torch
import triton
import triton.language as tl


# Pattern: match relu + dropout(identity, p=0, training=False) + flatten.
# The target graph uses torch.nn.functional.relu and torch.nn.functional.dropout.
# We use torch.nn.functional.relu and torch.dropout to match correctly.
def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _triton_relu_flatten_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


@torch.fx.wrap
def triton_relu_dropout_flatten(in_0):
    # Apply relu + flatten in one kernel (dropout with p=0,training=False is identity)
    batch = in_0.shape[0]
    n_elements = in_0.numel()
    n_channels = n_elements // batch
    out = torch.empty((batch, n_channels), dtype=in_0.dtype, device=in_0.device)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _triton_relu_flatten_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
    )
    return out


def replacement_func():
    return triton_relu_dropout_flatten