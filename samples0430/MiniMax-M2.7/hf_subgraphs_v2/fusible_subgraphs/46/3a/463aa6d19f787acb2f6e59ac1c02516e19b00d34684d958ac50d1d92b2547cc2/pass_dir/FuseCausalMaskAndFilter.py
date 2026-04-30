import torch
import triton
import triton.language as tl

NEG_INF = -3.4028234663852886e+38


@triton.jit
def fused_masked_fill_kernel(
    x_ptr,
    mask_ptr,
    output_ptr,
    n_elements,
    NEG_INF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for masked_fill operation."""
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    boundary_mask = offsets < n_elements
    
    # Load values
    x = tl.load(x_ptr + offsets, mask=boundary_mask, other=0.0)
    m = tl.load(mask_ptr + offsets, mask=boundary_mask, other=0.0)
    
    # Apply mask: if m is True (non-zero), set to NEG_INF
    result = tl.where(m != 0, NEG_INF, x)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=boundary_mask)


@torch.fx.wrap
def fused_masked_fill(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Optimized masked_fill that fuses comparison and fill."""
    n_elements = x.numel()
    
    # Output
    output = torch.empty_like(x)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_masked_fill_kernel[(num_programs,)](
        x_ptr=x,
        mask_ptr=mask,
        output_ptr=output,
        n_elements=n_elements,
        NEG_INF=NEG_INF,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(x, mask):
    """
    Match the masked_fill operation.
    The mask is expected to be the result of comparing with 0,
    but we match just the masked_fill to avoid operator mismatches.
    """
    return x.masked_fill(mask, NEG_INF)


def replacement_args(tmp_16, tmp_15):
    """Extract arguments for the replacement function."""
    return (tmp_16, tmp_15)


def replacement_func():
    """Return the optimized function."""
    return fused_masked_fill