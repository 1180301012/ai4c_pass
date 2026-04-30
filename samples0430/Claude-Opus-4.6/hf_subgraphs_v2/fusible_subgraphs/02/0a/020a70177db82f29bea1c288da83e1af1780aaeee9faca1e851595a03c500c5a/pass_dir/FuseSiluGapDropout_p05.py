import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.5, False, True)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def silu_gap_flatten_kernel(
    input_ptr,
    output_ptr,
    BC,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bc_idx = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW

    base_offset = bc_idx * HW
    x = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)

    # SiLU: x * sigmoid(x) in float32 for accuracy
    x_float = x.to(tl.float32)
    sigmoid_x = tl.sigmoid(x_float)
    silu_x = x_float * sigmoid_x

    # Global average pool
    sum_val = tl.sum(silu_x, axis=0)
    avg_val = sum_val / HW

    # Store result
    tl.store(output_ptr + bc_idx, avg_val.to(x.dtype))


@torch.fx.wrap
def silu_gap_flatten(in_0):
    B, C, H, W = in_0.shape
    HW = H * W
    BC = B * C

    output = torch.empty((B, C), dtype=in_0.dtype, device=in_0.device)

    BLOCK_SIZE = triton.next_power_of_2(HW)
    if BLOCK_SIZE < 64:
        BLOCK_SIZE = 64

    grid = (BC,)
    silu_gap_flatten_kernel[grid](
        in_0,
        output,
        BC,
        HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return silu_gap_flatten