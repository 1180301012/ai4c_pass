import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_relu_scale_bias_kernel(
    in2_ptr,
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load scalar bias and scale (broadcast across all elements)
    bias = tl.load(in0_ptr)
    scale = tl.load(in1_ptr)

    # Load input feature map
    x = tl.load(in2_ptr + offsets, mask=mask, other=0.0)

    # Fused computation: bias + scale * relu(x)
    relu_x = tl.maximum(x, 0.0)
    out = bias + scale * relu_x

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu_scale_bias(in_0, in_1, in_2):
    out = torch.empty_like(in_2)
    n_elements = in_2.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    fused_relu_scale_bias_kernel[grid](
        in2_ptr=in_2,
        in0_ptr=in_0,
        in1_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_relu_scale_bias