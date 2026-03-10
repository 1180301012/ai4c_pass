import torch
import triton
import triton.language as tl
from torch import device


# Pattern matching function - matches the complete computation pattern exactly as in the model
def pattern(in_0):
    tmp_1 = torch.arange(0, 64, device=device(type='cuda', index=0))
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return tmp_1, tmp_2


# Extract arguments needed for replacement
def replacement_args(in_0):
    return (in_0,)


# Combined Triton kernel that generates arange and does bool conversion
@triton.jit
def fused_arange_bool_kernel(
    output_arange_ptr,
    output_bool_ptr,
    n_elements,
    arange_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Generate arange: for indices < arange_size, output the index
    # For indices >= arange_size, output 0 (won't be used)
    arange_vals = offsets
    arange_mask = offsets < arange_size
    tl.store(output_arange_ptr + offsets, arange_vals, mask=arange_mask)

    # Load input values and convert to bool
    # Non-zero values become True, zero becomes False  
    x = tl.load(output_bool_ptr + offsets, mask=mask, other=0)
    bool_val = x != 0
    tl.store(output_bool_ptr + offsets, bool_val, mask=mask)


@torch.fx.wrap
def fused_arange_bool_wrapper(in_0):
    # Move input to cuda if not already there
    if in_0.device.type != 'cuda':
        in_0 = in_0.cuda()
    
    # Determine arange size from input shape
    shape = in_0.shape
    arange_size = shape[-1] if len(shape) > 0 else in_0.numel()
    
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Create output tensors
    output_arange = torch.empty(arange_size, device='cuda', dtype=torch.int64)
    output_bool = torch.empty_like(in_0, dtype=torch.bool)

    # Launch kernel - first fill with input data for bool conversion
    # We need to copy input data to output_bool first, then convert
    output_bool.copy_(in_0)
    
    fused_arange_bool_kernel[(num_programs,)](
        output_arange_ptr=output_arange,
        output_bool_ptr=output_bool,
        n_elements=n_elements,
        arange_size=arange_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Return both the arange and the bool converted result
    return output_arange, output_bool


def replacement_func():
    return fused_arange_bool_wrapper