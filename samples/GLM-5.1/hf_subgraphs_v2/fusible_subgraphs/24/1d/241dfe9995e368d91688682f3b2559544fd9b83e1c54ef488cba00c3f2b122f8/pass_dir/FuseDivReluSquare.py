import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0 / 11.313708498984761
    tmp_1 = torch.relu(tmp_0)
    tmp_2 = torch.pow(tmp_1, 2)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_div_relu_square_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # Divide by 11.313708498984761 (multiply by reciprocal for performance)
    # reciprocal = 1.0 / 11.313708498984761 ≈ 0.08838834764831845
    scale_recip = 0.08838834764831845
    x = x * scale_recip

    # ReLU: max(0, x)
    x = tl.maximum(x, 0.0)

    # Square
    x = x * x

    # Store
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_div_relu_square(in_0):
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in_0)

    fused_div_relu_square_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_div_relu_square