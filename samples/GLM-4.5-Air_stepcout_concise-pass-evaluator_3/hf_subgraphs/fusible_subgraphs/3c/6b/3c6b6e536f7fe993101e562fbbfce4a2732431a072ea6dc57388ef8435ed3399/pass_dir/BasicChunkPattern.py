import torch
import triton
import triton.language as tl

def pattern(in_3):
    tmp = in_3.chunk(2, dim=-1)
    a = tmp[0]
    b = tmp[1]
    return a, b

def replacement_args(in_3):
    return (in_3,)

@triton.jit
def basic_chunk_kernel(
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
    
    # For chunking along last dimension with 2 chunks
    # Each element goes to either out1 or out2 based on position in last dimension
    last_dim = offsets % 32  # Assuming split happens at dimension 32
    
    mask1 = last_dim < 16  # First half of last dimension
    mask2 = last_dim >= 16  # Second half of last dimension
    
    if tl.any(mask1):
        tl.store(out1_ptr + offsets[mask1], x[mask1], mask=mask1)
    if tl.any(mask2):
        tl.store(out2_ptr + offsets[mask2], x[mask2], mask=mask2)

@torch.fx.wrap
def basic_chunk_optimization(in_3):
    # Get input shape
    original_shape = in_3.shape
    last_dim = original_shape[-1]
    
    if last_dim % 2 != 0:
        # If not divisible by 2, fall back to original
        chunked = in_3.chunk(2, dim=-1)
        return chunked[0], chunked[1]
    
    chunk_shape = list(original_shape[:-1]) + [last_dim // 2]
    
    out1 = torch.empty(chunk_shape, dtype=in_3.dtype, device=in_3.device)
    out2 = torch.empty(chunk_shape, dtype=in_3.dtype, device=in_3.device)
    
    n_elements = in_3.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    basic_chunk_kernel[(num_programs,)](
        in_ptr=in_3,
        out1_ptr=out1,
        out2_ptr=out2,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1, out2

def replacement_func():
    return basic_chunk_optimization