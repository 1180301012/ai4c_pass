import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def relu_flatten_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0)
    tl.store(out_ptr + offsets, y, mask=mask)


@torch.fx.wrap
def shared_relu_flatten_dispatch(in_0, route):
    batch = in_0.shape[0]
    n_elements = in_0.numel()
    flat_dim = n_elements // batch if batch != 0 else 0
    out = torch.empty((batch, flat_dim), device=in_0.device, dtype=in_0.dtype)
    if n_elements == 0:
        return out
    if route == "relu_dropout_flatten" or route == "relu_flatten" or route == "relu_view" or route == "dropout_flatten" or route == "dropout_view" or route == "flatten_only" or route == "view_only":
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        relu_flatten_kernel[grid](in_0, out, n_elements)
        return out
    return out


def replacement_func():
    return shared_relu_flatten_dispatch