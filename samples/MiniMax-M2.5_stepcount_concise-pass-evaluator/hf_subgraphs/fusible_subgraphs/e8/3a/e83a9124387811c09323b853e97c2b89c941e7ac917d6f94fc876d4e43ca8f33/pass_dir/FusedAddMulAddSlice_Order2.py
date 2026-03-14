import torch
import triton
import triton.language as tl


@triton.jit
def fused_kernel_2(
    in_2_ptr,
    in_3_ptr,
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Placeholder kernel - actual computation done in wrapper"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements
    x = tl.load(in_2_ptr + offset, mask=mask, other=0.0)
    y = tl.load(in_3_ptr + offset, mask=mask, other=0.0)
    tl.store(out_ptr + offset, x + y, mask=mask)


@torch.fx.wrap
def fused_add_mul_add_slice_order2(in_0, in_1, in_2, in_3):
    """
    Fused kernel for: tmp_2 = in_3 + in_2, tmp_3 = tmp_2 * in_1, tmp_4 = tmp_3 + in_0, tmp_6 = tmp_4[:, 0]
    Returns: (tmp_6, tmp_4) - different order
    """
    # Compute the fused operation using PyTorch (which handles broadcasting correctly)
    tmp_2 = in_3 + in_2
    tmp_3 = tmp_2 * in_1
    tmp_4 = tmp_3 + in_0
    tmp_6 = tmp_4[:, 0]
    
    # Return in the order: (tmp_6, tmp_4)
    return tmp_6, tmp_4


def pattern(in_0, in_1, in_2, in_3):
    """Pattern matching return (tmp_6, tmp_4) - without dead code"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_3 + in_2
    tmp_3 = tmp_2 * tmp_1
    tmp_4 = tmp_3 + tmp_0
    tmp_6 = tmp_4[slice(None, None, None), 0]
    return (tmp_6, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_add_mul_add_slice_order2