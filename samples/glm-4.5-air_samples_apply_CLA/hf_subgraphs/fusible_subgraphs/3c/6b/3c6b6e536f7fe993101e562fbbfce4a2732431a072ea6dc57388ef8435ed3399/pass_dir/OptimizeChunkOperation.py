import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_3):
    # Chunk operation pattern:
    # tmp_8 = in_3.chunk(2, dim=-1)
    # tmp_9 = tmp_8[0]
    # tmp_10 = tmp_8[1]
    
    tmp_8 = in_3.chunk(2, dim=-1)
    tmp_9 = tmp_8[0]
    tmp_10 = tmp_8[1]
    return tmp_9, tmp_10

# Argument extraction function
def replacement_args(in_3):
    return (in_3,)

# Optimized Triton kernel for chunk operation
@triton.jit
def chunk_kernel(
    in_3_ptr,
    out_9_ptr, out_10_ptr,
    batch_size, seq_len, in_3_d2, in_3_d3,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate offset for processing (treat as flattened tensor)
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total elements for half the last dimension
    total_elements_half = batch_size * seq_len * (in_3_d3 // 2)
    mask = offset < total_elements_half
    
    # Load input tensor (since we're doing chunk, we need to load the full tensor first)
    # But actually, we'll compute the chunk directly by indexing
    base_offset = offset
    
    # For chunk operation, we need to split along last dimension
    # Element i in out_9 corresponds to element i in input
    # Element i in out_10 corresponds to element i + half_elements in input
    in_9_val = tl.load(in_3_ptr + base_offset, mask=mask, other=0.0)
    in_10_val = tl.load(in_3_ptr + base_offset + total_elements_half, mask=mask, other=0.0)
    
    # Store both chunks
    tl.store(out_9_ptr + base_offset, in_9_val, mask=mask)
    tl.store(out_10_ptr + base_offset, in_10_val, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def chunk_fused(in_3):
    # Get input shapes
    batch_size = in_3.shape[0]
    seq_len = in_3.shape[1] if len(in_3.shape) > 1 else 1
    in_3_d2 = in_3.shape[2] if len(in_3.shape) > 2 else 1
    in_3_d3 = in_3.shape[3] if len(in_3.shape) > 3 else 1
    
    # Calculate total elements for half the last dimension
    total_elements_half = batch_size * seq_len * (in_3_d3 // 2)
    
    # Create output tensors for both chunks
    out_9 = torch.empty((batch_size, seq_len, in_3_d2, in_3_d3 // 2), dtype=torch.float32, device=in_3.device)
    out_10 = torch.empty((batch_size, seq_len, in_3_d2, in_3_d3 // 2), dtype=torch.float32, device=in_3.device)
    
    # Configure grid and block size
    BLOCK_SIZE = 1024
    num_programs = (total_elements_half + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    chunk_kernel[(num_programs,)](
        in_3_ptr=in_3,
        out_9_ptr=out_9,
        out_10_ptr=out_10,
        batch_size=batch_size,
        seq_len=seq_len,
        in_3_d2=in_3_d2,
        in_3_d3=in_3_d3,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_9, out_10

# Replacement function
def replacement_func():
    return chunk_fused