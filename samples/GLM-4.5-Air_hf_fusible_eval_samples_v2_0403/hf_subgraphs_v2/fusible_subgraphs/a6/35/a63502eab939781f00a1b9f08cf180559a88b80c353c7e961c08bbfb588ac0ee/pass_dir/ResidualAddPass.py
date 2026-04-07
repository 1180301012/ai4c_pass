import torch
import triton
import triton.language as tl

@triton.jit
def residual_add_kernel(
    residual_ptr,
    norm_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for residual addition: residual + normalized."""
    pid = tl.program_id(0)
    total_elements = n_elements
    block_size = BLOCK_SIZE
    
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Load residual and normalized tensors
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    normalized = tl.load(norm_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    out = residual + normalized
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_residual_add(residual, normalized):
    """Residual addition using Triton."""
    N = residual.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(residual)

    residual_add_kernel[(num_programs,)](
        residual_ptr=residual,
        norm_ptr=normalized,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def pattern(in_2, tmp_6):
    """Pattern matches the residual addition from the model."""
    tmp_7 = in_2 + tmp_6
    return in_2, tmp_7

def replacement_args(in_2, tmp_6):
    return (in_2, tmp_6)

def replacement_func():
    def residual_add_wrapper(in_2, tmp_6):
        return triton_residual_add(in_2, tmp_6)
    return residual_add_wrapper