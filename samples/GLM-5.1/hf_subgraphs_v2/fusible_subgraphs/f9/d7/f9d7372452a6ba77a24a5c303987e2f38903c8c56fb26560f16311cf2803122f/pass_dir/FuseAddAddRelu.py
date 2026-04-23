import torch
import triton
import triton.language as tl


def pattern(in_0, in_2, in_3):
    # Match: in_3 += in_0 => in_4 = in_3 + in_0
    # Match: in_4 += in_2 => tmp_0 = in_4 + in_2
    # Match: relu(tmp_0, inplace=True) => tmp_2
    in_4 = in_3 + in_0
    tmp_0 = in_4 + in_2
    tmp_2 = torch.nn.functional.relu(tmp_0, inplace=True)
    return tmp_2


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)


@triton.jit
def fused_add_add_relu_kernel(
    in_0_ptr,
    in_2_ptr,
    in_3_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)

    # Fused: relu(in_0 + in_2 + in_3)
    result = in_0 + in_2 + in_3
    result = tl.maximum(result, 0.0)

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_add_add_relu(in_0, in_2, in_3):
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in_0)

    fused_add_add_relu_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_add_add_relu