import torch
from torch import device
import triton
import triton.language as tl


def pattern():
    tmp_0 = torch.arange(0, 1, device=device(type='cuda', index=0))
    tmp_1 = tmp_0.unsqueeze(0)
    tmp_2 = tmp_1.repeat(1, 1)
    return tmp_0, tmp_2


def replacement_args():
    return ()


@triton.jit
def fused_arange_unsqueeze_repeat_kernel(
    out_ptr,
    n_elements,
    start: tl.constexpr,
    end: tl.constexpr,
    repeat_dims_0: tl.constexpr,
    repeat_dims_1: tl.constexpr,
    output_shape_0: tl.constexpr,
    output_shape_1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate output indices from linear offset
    out_idx_0 = offsets // output_shape_1
    out_idx_1 = offsets % output_shape_1

    # Calculate input indices (inverse of repeat operation)
    in_idx_0 = out_idx_0 % repeat_dims_0
    in_idx_1 = out_idx_1 % repeat_dims_1

    # Calculate linear index into source (after unsqueeze, source has shape [1])
    # Since unsqueeze adds dimension at 0, and source is arange(start, end),
    # the value at position [i] is i + start
    linear_idx = in_idx_0 * 1 + in_idx_1

    # Clamp to valid range
    val = linear_idx + start
    val = tl.where(val < end, val, end - 1)
    val = tl.where(val >= start, val, start)

    tl.store(out_ptr + offsets, val.to(tl.float16), mask=mask)


@torch.fx.wrap
def fused_arange_unsqueeze_repeat_wrapper():
    start = 0
    end = 1
    repeat_dims = (1, 1)
    output_shape = (1, 1)  # After unsqueeze(0), arange(0, 1) has shape (1,), then repeat(1, 1)

    out_ptr = torch.empty(output_shape, dtype=torch.float16, device='cuda')
    n_elements = out_ptr.numel()
    BLOCK_SIZE = 1
    num_programs = 1

    fused_arange_unsqueeze_repeat_kernel[(num_programs,)](
        out_ptr=out_ptr,
        n_elements=n_elements,
        start=start,
        end=end,
        repeat_dims_0=repeat_dims[0],
        repeat_dims_1=repeat_dims[1],
        output_shape_0=output_shape[0],
        output_shape_1=output_shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Create tmp_0: the original arange output (shape [1])
    tmp_0 = torch.arange(start, end, device='cuda', dtype=torch.float16)
    # tmp_2 is the repeated tensor
    tmp_2 = out_ptr

    return tmp_0, tmp_2


def replacement_func():
    return fused_arange_unsqueeze_repeat_wrapper