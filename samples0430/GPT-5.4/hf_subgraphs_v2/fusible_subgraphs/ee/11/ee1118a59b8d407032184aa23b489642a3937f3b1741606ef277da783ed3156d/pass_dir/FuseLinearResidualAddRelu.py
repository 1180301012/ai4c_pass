import torch
import triton
import triton.language as tl


def pattern(in_2, linear):
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return tmp_4


def replacement_args(in_2, linear):
    return (in_2, linear)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_add_relu_kernel(
    residual_ptr,
    linear_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    linear = tl.load(linear_ptr + offsets, mask=mask, other=0.0)
    out = residual + linear
    out = tl.maximum(out, 0.0)
    tl.store(linear_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_linear_residual_relu(in_2, linear):
    n_elements = linear.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    fused_add_relu_kernel[grid](
        in_2,
        linear,
        n_elements,
    )
    return linear


def replacement_func():
    return fused_linear_residual_relu