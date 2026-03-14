import torch
import triton
import triton.language as tl


def pattern(in_6, in_5, in_2, in_4):
    """
    Pattern to match:
    tmp_0 = -in_6
    tmp_1 = torch.cat((tmp_0, in_5), dim=-1)
    tmp_2 = tmp_1 * in_2
    tmp_3 = in_4 + tmp_2
    tmp_4 = tmp_3.to(dtype=torch.float32)
    """
    tmp_0 = -in_6
    tmp_1 = torch.cat((tmp_0, in_5), dim=-1)
    tmp_2 = tmp_1 * in_2
    tmp_3 = in_4 + tmp_2
    tmp_4 = tmp_3.to(dtype=torch.float32)
    return tmp_4


def replacement_args(in_6, in_5, in_2, in_4):
    return (in_6, in_5, in_2, in_4)


@triton.jit
def fused_neg_cat_mul_add_cast_kernel(
    in_6_ptr,
    in_5_ptr,
    in_2_ptr,
    in_4_ptr,
    out_ptr,
    n_elements,
    half_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: cat(-in_6, in_5) * in_2 + in_4, cast to float32
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load from in_2 and in_4
    in_2_val = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_4_val = tl.load(in_4_ptr + offsets, mask=mask, other=0.0)
    
    # Compute which part of concatenation we're in
    full_size = half_size * 2
    col = offsets % full_size  # position in last dimension
    row = offsets // full_size  # "row number" (all dims except last)
    
    # Compute input index
    is_first_half = col < half_size
    input_col = tl.where(is_first_half, col, col - half_size)
    input_idx = row * half_size + input_col
    
    # Load only the value we need based on which half
    concat_val = tl.zeros_like(in_2_val)
    
    # For first half: load from in_6 and negate
    in_6_val = tl.load(in_6_ptr + input_idx, mask=mask & is_first_half, other=0.0)
    concat_val = tl.where(is_first_half, -in_6_val, concat_val)
    
    # For second half: load from in_5
    in_5_val = tl.load(in_5_ptr + input_idx, mask=mask & (~is_first_half), other=0.0)
    concat_val = tl.where(~is_first_half, in_5_val, concat_val)
    
    # Compute result
    result = in_4_val + concat_val * in_2_val
    
    # Store
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_neg_cat_mul_add_cast(in_6, in_5, in_2, in_4):
    """
    Fused implementation using Triton kernel
    """
    output = torch.empty_like(in_4, dtype=torch.float32)
    n_elements = in_4.numel()
    half_size = in_6.shape[-1]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_neg_cat_mul_add_cast_kernel[grid](
        in_6,
        in_5,
        in_2,
        in_4,
        output,
        n_elements,
        half_size,
        BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_neg_cat_mul_add_cast