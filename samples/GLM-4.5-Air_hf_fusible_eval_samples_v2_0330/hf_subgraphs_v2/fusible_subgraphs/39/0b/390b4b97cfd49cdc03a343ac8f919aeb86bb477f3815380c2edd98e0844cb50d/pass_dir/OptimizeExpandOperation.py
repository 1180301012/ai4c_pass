import torch
import triton
import triton.language as tl

# Pattern matching function - unsqueeze followed by expand
def pattern(tmp_2):
    tmp_5 = tmp_2.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    return tmp_6

# Argument extraction function
def replacement_args(tmp_2):
    return (tmp_2,)

# Optimized kernel - direct expansion without intermediate unsqueeze
@triton.jit
def direct_expand_3d_kernel(
    tmp_2_ptr,
    out_ptr,
    n_elements,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * n_elements)
    
    # Determine which batch and which element in batch
    batch_idx = offsets // n_elements
    elem_idx = offsets % n_elements
    
    # Load input data - note we only load from the original 2D tensor
    tmp_2_data = tl.load(tmp_2_ptr + elem_idx, mask=elem_idx < n_elements, other=0)
    
    # Broadcast to all 3 batches by repeating the loaded value
    # Since all batches have the same data, we just need to copy
    out_data = tmp_2_data
    
    # Store result - each position gets the same repeated value
    tl.store(out_ptr + offsets, out_data, mask=mask)

# Wrapper function
@torch.fx.wrap
def direct_expand_3d(tmp_2):
    original_shape = tmp_2.shape
    expanded_shape = (3,) + original_shape  # Expand to 3 batches
    batch_size = 3
    n_elements = tmp_2.numel()
    
    BLOCK_SIZE = 1024
    num_programs = (batch_size * n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(expanded_shape, dtype=tmp_2.dtype, device=tmp_2.device)
    
    direct_expand_3d_kernel[(num_programs,)](
        tmp_2_ptr=tmp_2,
        out_ptr=out,
        n_elements=n_elements,
        batch_size=batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return direct_expand_3d