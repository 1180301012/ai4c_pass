import torch
import triton
import triton.language as tl


# Match the observable subgraph exactly: add followed by in-place relu.
def pattern(in_2, linear):
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return tmp_4


# Preserve input ordering from the target graph: residual first, linear second.
def replacement_args(in_2, linear):
    return (in_2, linear)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _add_relu_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    out = tl.maximum(out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_relu(in_2, linear):
    out = torch.empty_like(in_2)
    n_elements = out.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    _add_relu_kernel[grid](
        in_2,
        linear,
        out,
        n_elements,
    )
    return out


def replacement_func():
    return fused_add_relu