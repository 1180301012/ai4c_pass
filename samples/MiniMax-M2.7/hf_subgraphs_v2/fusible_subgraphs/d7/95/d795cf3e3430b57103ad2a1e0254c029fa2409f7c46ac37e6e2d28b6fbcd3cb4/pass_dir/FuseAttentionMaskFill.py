import torch
import triton
import triton.language as tl


@triton.jit  
def fused_comparison_and_fill_kernel(
    data_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully fused kernel: compute comparison + two masked_fill operations.
    Computes: where(data == 0, 0.0, -1000.0)
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = tl.load(data_ptr + offsets, mask=mask, other=0.0)
    
    # Fused comparison + fill: where(data == 0, 0.0, -1000.0)
    result = tl.where(data == 0, 0.0, -1000.0)
    
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_mask_fill_wrapper(data, mask_ne, mask_eq):
    """
    Wrapper that uses Triton kernel to fuse:
    tmp_14 = data.masked_fill(mask_ne, -1000.0)
    tmp_16 = tmp_14.masked_fill(mask_eq, 0.0)
    
    Into a single kernel: where(data == 0, 0.0, -1000.0)
    Note: mask_ne = (data != 0), mask_eq = (data == 0)
    """
    N = data.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(data)
    
    fused_comparison_and_fill_kernel[(num_programs,)](
        data_ptr=data,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(data, mask_ne, mask_eq):
    """
    Match the pattern:
    tmp_14 = data.masked_fill(mask_ne, -1000.0)
    tmp_16 = tmp_14.masked_fill(mask_eq, 0.0)
    """
    tmp_14 = data.masked_fill(mask_ne, -1000.0)
    tmp_16 = tmp_14.masked_fill(mask_eq, 0.0)
    return tmp_16


def replacement_args(data, mask_ne, mask_eq):
    return (data, mask_ne, mask_eq)


def replacement_func():
    return fused_mask_fill_wrapper