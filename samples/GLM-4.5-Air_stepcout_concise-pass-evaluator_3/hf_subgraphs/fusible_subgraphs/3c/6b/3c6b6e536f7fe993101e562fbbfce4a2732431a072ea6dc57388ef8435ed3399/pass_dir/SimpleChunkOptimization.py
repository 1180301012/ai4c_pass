import torch
import triton
import triton.language as tl

def pattern(in_3):
    x = in_3.chunk(2, dim=-1)
    y = x[0]
    z = x[1]
    return y, z

def replacement_args(in_3):
    return (in_3,)

@triton.jit
def simple_chunk_kernel(
    in_ptr,
    out1_ptr,
    out2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # For chunking, we need to split the last dimension
    # Assume the chunk operation splits the last dimension in half
    # Each program will process elements and determine if they go to out1 or out2
    half_point = n_elements // 2
    
    # Create masks for each chunk
    mask1 = offsets < half_point
    mask2 = offsets >= half_point
    
    # Process first chunk
    if tl.any(mask1):
        # Calculate indices for first chunk (0 to half_point-1)
        active_indices = offsets[mask1]
        chunk_indices = active_indices
        tl.store(out1_ptr + chunk_indices, x[mask1], mask=mask1)
    
    # Process second chunk  
    if tl.any(mask2):
        # Calculate indices for second chunk (half_point to end)
        active_indices = offsets[mask2] - half_point
        tl.store(out2_ptr + active_indices, x[mask2], mask=mask2)

@torch.fx.wrap
def simple_chunk_optimization(in_3):
    # Get input shape
    original_shape = in_3.shape
    last_dim = original_shape[-1]
    
    # Check if last dimension is divisible by 2
    if last_dim % 2 != 0:
        # If not, we can't chunk properly, return original
        chunked = in_3.chunk(2, dim=-1)
        return chunked[0], chunked[1]
    
    # Each chunk has half the elements in last dimension
    chunk_shape = list(original_shape[:-1]) + [last_dim // 2]
    
    out1 = torch.empty(chunk_shape, dtype=in_3.dtype, device=in_3.device)
    out2 = torch.empty(chunk_shape, dtype=in_3.dtype, device=in_3.device)
    
    # Total elements in original tensor
    n_elements = in_3.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_chunk_kernel[(num_programs,)](
        in_ptr=in_3,
        out1_ptr=out1,
        out2_ptr=out2,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1, out2

def replacement_func():
    return simple_chunk_optimization