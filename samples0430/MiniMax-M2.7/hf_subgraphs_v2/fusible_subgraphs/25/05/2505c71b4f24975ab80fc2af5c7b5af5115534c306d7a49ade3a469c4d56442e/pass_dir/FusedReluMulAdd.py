import torch
import triton
import triton.language as tl


@triton.jit
def fused_relu_mul_add_kernel(
    x_ptr,
    scale_val,
    bias_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: relu(x) * scale + bias
    scale_val and bias_val are scalar values loaded externally
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Fused ops: relu(x) * scale + bias (scale and bias are scalars)
    x_relu = tl.where(x > 0, x, 0.0)
    out = x_relu * scale_val + bias_val

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu_mul_add(x, scale, bias):
    """
    Wrapper for the fused relu * scale + bias kernel.
    x: input tensor [N, C, H, W]
    scale: scalar tensor [1]
    bias: scalar tensor [1]
    """
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    # Extract scalar values from scale and bias tensors
    scale_val = scale.item() if scale.numel() == 1 else scale
    bias_val = bias.item() if bias.numel() == 1 else bias

    fused_relu_mul_add_kernel[(num_programs,)](
        x_ptr=x,
        scale_val=scale_val,
        bias_val=bias_val,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: relu(in_2) * in_1 + in_0
    where in_0 is bias, in_1 is scale, in_2 is input
    """
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_relu_mul_add