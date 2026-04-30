import torch
import triton
import triton.language as tl


# Match exactly: relu(inplace=True) followed by dropout2d(..., training=False, inplace=False)
# Both outputs are observable, so both must be returned.
def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)


# Only the input tensor is needed because dropout2d is an identity in eval mode.
def replacement_args(in_0):
    return (in_0,)


@triton.jit
def _relu_inplace_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0)
    tl.store(x_ptr + offsets, y, mask=mask)


@torch.fx.wrap
def relu_inplace_dropout2d_eval(x):
    n_elements = x.numel()
    if n_elements == 0:
        return (x, x)

    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _relu_inplace_kernel[grid](
        x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=4,
    )
    return (x, x)


# Must return the callable, not call it.
def replacement_func():
    return relu_inplace_dropout2d_eval