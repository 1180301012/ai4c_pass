import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, x, 0.0)
    tl.store(output_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu_dropout_eval(input_tensor):
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    output = torch.empty_like(input_tensor)

    relu_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Since dropout2d with training=False is identity, both outputs are the relu result
    return (output, output)


def replacement_func():
    return fused_relu_dropout_eval