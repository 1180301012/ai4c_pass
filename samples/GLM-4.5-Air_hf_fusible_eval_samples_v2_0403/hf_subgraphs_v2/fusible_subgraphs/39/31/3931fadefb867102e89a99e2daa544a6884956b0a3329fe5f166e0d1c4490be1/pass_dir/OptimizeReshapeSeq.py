import torch
import triton
import triton.language as tl

# Pattern matching function - match the two consecutive reshape operations
def pattern(tmp_1):
    # tmp_1 has shape [1, 124, 2, 768]
    tmp_2 = tmp_1.reshape(1, 248, 768)
    return tmp_2

# Argument extraction function
def replacement_args(tmp_1):
    return (tmp_1,)

# Triton kernel that eliminates the intermediate reshape
@triton.jit
def reshape_elimination_kernel(
    in_ptr,          # Input tensor pointer [1, 124, 2, 768]
    out_ptr,         # Output tensor pointer [1, 248, 768]
    in_elements,     # Total input elements: 1 * 124 * 2 * 768
    out_elements,    # Total output elements: 1 * 248 * 768
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID determines which block of data this program handles
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < in_elements
    
    # Load input data (same shape as output, so direct copy)
    data = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # The key insight: reshape [1, 124, 2, 768] -> [1, 248, 768] 
    # is equivalent to just changing how we interpret the coordinates
    # since the total number of elements is the same
    
    # Direct coordinate mapping without any computation:
    # Input coords: [1, 124, 2, 768] -> flatten to 1D
    # Output coords: [1, 248, 768] -> flatten to 1D
    
    # For elements in the same position within each [2, 768] block,
    # we just need to adjust the sequence dimension
    
    # Convert 1D index to original 4D coordinates
    total_seq_blocks = 124 * 2  # Each original sequence becomes 2 sequences
    elements_per_seq = 768       # Elements per sequence
    
    sequence_group = offsets // (total_seq_blocks * elements_per_seq)  # Always 0
    local_offset = offsets % (total_seq_blocks * elements_per_seq)
    new_seq = local_offset // elements_per_seq    # 0-247 instead of 0-123 then 0-1
    elem_pos = local_offset % elements_per_seq    # 0-767
    
    # Calculate new 1D index: [1, 248, 768] -> 1D
    out_idx = sequence_group * (248 * 768) + new_seq * 768 + elem_pos
    
    # Bounds check
    valid_mask = mask & (out_idx < out_elements)
    
    # Store directly - this is essentially a memory copy
    # but optimized by eliminating the reshape operation
    tl.store(out_ptr + out_idx, data, mask=valid_mask)

# Simple copy wrapper for the reshape elimination
@torch.fx.wrap
def optimize_reshape_sequence(tmp_1):
    # Input shape: [1, 124, 2, 768]
    # Output shape: [1, 248, 768]
    in_elements = 1 * 124 * 2 * 768  # 190464
    out_elements = 1 * 248 * 768     # 190464 (same number of elements)
    
    # Use optimal block size for memory copy operations
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (in_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty([1, 248, 768], dtype=tmp_1.dtype, device=tmp_1.device)
    
    # Launch reshape elimination kernel
    reshape_elimination_kernel[(num_programs,)](
        in_ptr=tmp_1,
        out_ptr=out,
        in_elements=in_elements,
        out_elements=out_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return optimize_reshape_sequence