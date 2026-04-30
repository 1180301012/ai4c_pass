import torch
import triton
import triton.language as tl


@triton.jit
def fused_masked_fill_kernel(
    masked_1_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that combines:
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    
    Where tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0) with tmp_13 = tmp_12 != 0
    and tmp_15 = tmp_12 == 0
    
    The kernel applies the mask and computes the final result.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    masked_1 = tl.load(masked_1_ptr + offsets, mask=mask, other=0.0)
    mask_val = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    
    # tmp_16 = tmp_14.masked_fill(tmp_15, 0.0) = where(tmp_15, 0.0, tmp_14)
    result = tl.where(mask_val, 0.0, masked_1)
    
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_masked_fill(masked_1: torch.Tensor, mask_val: torch.Tensor) -> torch.Tensor:
    """
    Fused implementation of:
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    
    Returns tmp_16.
    """
    N = masked_1.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(masked_1)

    fused_masked_fill_kernel[(num_programs,)](
        masked_1_ptr=masked_1,
        mask_ptr=mask_val,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def pattern(tmp_14, tmp_15):
    """
    Match the pattern:
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    """
    result = tmp_14.masked_fill(tmp_15, 0.0)
    return result


def replacement_args(tmp_14, tmp_15):
    return (tmp_14, tmp_15)


def replacement_func():
    return fused_masked_fill