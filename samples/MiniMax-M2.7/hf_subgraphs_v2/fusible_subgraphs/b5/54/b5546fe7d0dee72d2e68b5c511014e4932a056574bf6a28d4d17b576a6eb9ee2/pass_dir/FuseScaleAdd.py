import torch
import triton
import triton.language as tl


@triton.jit
def triton_scale_add_kernel(
    in_4_ptr,
    tmp_7_ptr,
    out_ptr,
    n_elements,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: scale * in_4 + tmp_7 (element-wise)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values at flat offset
    in_4_val = tl.load(in_4_ptr + offsets, mask=mask, other=0.0)
    tmp_7_val = tl.load(tmp_7_ptr + offsets, mask=mask, other=0.0)

    # Compute scale * in_4 + tmp_7
    result = in_4_val * scale + tmp_7_val

    # Store result at flat offset
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def triton_scale_add(in_4, tmp_7, scale):
    """
    Fused scale and add operation.
    
    Args:
        in_4: Tensor
        tmp_7: Tensor (same shape as in_4)
        scale: Scalar constant
    
    Returns:
        Tensor with same shape as inputs
    """
    n_elements = in_4.numel()
    
    # Allocate output
    out = torch.empty_like(in_4)
    
    # Define block size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel - using flat indexing
    triton_scale_add_kernel[(num_programs,)](
        in_4,
        tmp_7,
        out,
        n_elements,
        scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


# Pattern matching function - matches the scale + add pattern
def pattern(in_4, tmp_7):
    """
    Match the pattern:
    tmp_8 = scale * in_4
    tmp_9 = tmp_8 + tmp_7
    
    The transpose and reshape will be handled by PyTorch.
    """
    c = 0.22941573387056177
    tmp_8 = c * in_4
    tmp_9 = tmp_8 + tmp_7
    return tmp_9


def replacement_args(in_4, tmp_7):
    """
    Extract arguments needed for the replacement kernel.
    """
    scale = 0.22941573387056177
    return (in_4, tmp_7, scale)


def replacement_func():
    """
    Return the replacement function that implements the fused kernel.
    """
    return triton_scale_add