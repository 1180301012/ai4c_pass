import torch
import triton
import triton.language as tl

@triton.jit
def fuse_div_add_add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: out = (x / 8.0) + y + z
    where z has shape [2, 1, 1, 7] broadcasting to [2, 12, 7, 7]
    """
    # Calculate program ID for each program
    pid = tl.program_id(axis=0)
    
    # Calculate offsets for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks
    mask = offsets < n_elements
    
    # Load x (attention_scores)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load y (values_2)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Load z (extended_attention_mask_2) - broadcasting handled by caller
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: (x / 8.0) + y + z
    # Division by 8.0, then two additions
    tmp = x / 8.0
    tmp = tmp + y
    out = tmp + z
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fuse_div_add_add_wrapper(x, y, z):
    """
    Wrapper function that launches the fused Triton kernel.
    Handles broadcasting of z from [2, 1, 1, 7] to [2, 12, 7, 7].
    """
    # Broadcast z to match x's shape
    # z is [2, 1, 1, 7], x is [2, 12, 7, 7]
    # We need to broadcast z to match x
    z_broadcast = z.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    
    # Total number of elements
    N = x.numel()
    
    # Define block size - use 1024 for good occupancy
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fuse_div_add_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z_broadcast,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: ((in_0 / 8.0) + in_2) + in_1
    Returns the final output tmp_2
    """
    tmp_0 = in_0 / 8.0
    tmp_0 = tmp_0 + in_2
    tmp_2 = tmp_0 + in_1
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement.
    in_0: attention_scores [2, 12, 7, 7]
    in_1: extended_attention_mask_2 [2, 1, 1, 7] 
    in_2: values_2 [2, 12, 7, 7]
    """
    return (in_0, in_1, in_2)


def replacement_func():
    """
    Return the wrapper function that launches the fused kernel.
    """
    return fuse_div_add_add_wrapper