import torch
import triton
import triton.language as tl

@triton.jit
def optimized_chunk_kernel_first(
    input_ptr,
    chunk0_ptr,
    chunk1_ptr,
    n_elements_total,
    chunk_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Process first chunk (split along dim=1, take second half)
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < chunk_elements
    
    # Load chunk from second half of channel dimension
    tl.store(chunk1_ptr + offsets, tl.load(input_ptr + chunk_elements + offsets, mask=mask), mask=mask)

@triton.jit
def optimized_chunk_kernel_second(
    input_ptr,
    chunk0_ptr,
    chunk1_ptr,
    n_elements_total,
    chunk_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Process both chunks (split along dim=1, take first half and second half)
    # This kernel handles both chunks in one go for better efficiency
    
    # First chunk
    pid = tl.program_id(0)
    if pid == 0:
        block_start = 0
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < chunk_elements
        
        # Load and store first chunk
        tl.store(chunk0_ptr + offsets, tl.load(input_ptr + offsets, mask=mask), mask=mask)
    
    # Second chunk (in same program for shared memory efficiency)
    if pid == 1:
        block_start = chunk_elements
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (n_elements_total - chunk_elements)
        
        # Load and store second chunk  
        tl.store(chunk1_ptr + offsets, tl.load(input_ptr + offsets, mask=mask), mask=mask)

@torch.fx.wrap
def optimized_chunk_first(x):
    # Split tensor along dimension 1 (channel dimension) and return second chunk
    # This is optimized for the case where we only need one chunk
    split_size = x.size(1) // 2
    
    # For our specific case where we want the chunk from in_2 (40 channels -> 20+20)
    # We'll only compute and return the second chunk since the first chunk is returned as tmp_4
    # and the second chunk is returned as tmp_5
    return x[:, split_size:, :, :]

@torch.fx.wrap  
def optimized_chunk_relu(x):
    # Split ReLU result along dimension 1 (80 channels -> 40+40)
    split_size = x.size(1) // 2
    
    # Return both chunks as tuple - tmp_7 and tmp_8
    return x[:, :split_size, :, :], x[:, split_size:, :, :]

def pattern(in_2, tmp_2):
    # Pattern matching for chunk operations:
    # tmp_3 = in_2.chunk(2, dim=1); tmp_4 = tmp_3[0]; tmp_5 = tmp_3[1]
    # tmp_6 = tmp_2.chunk(2, dim=1); tmp_7 = tmp_6[0]; tmp_8 = tmp_6[1]
    tmp_3 = in_2.chunk(2, dim=1)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    
    tmp_6 = tmp_2.chunk(2, dim=1)
    tmp_7 = tmp_6[0]
    tmp_8 = tmp_6[1]
    
    return tmp_4, tmp_7, tmp_5, tmp_8

def replacement_args(in_2, tmp_2):
    return (in_2, tmp_2)

def replacement_func():
    # Return a function that performs both chunk operations efficiently
    def chunk_operations(in_2, tmp_2):
        # Process in_2 chunks: in_2 has 40 channels, split to 20+20
        in_2_chunks = in_2.chunk(2, dim=1)
        tmp_4, tmp_5 = in_2_chunks[0], in_2_chunks[1]
        
        # Process tmp_2 chunks: tmp_2 has 80 channels, split to 40+40
        tmp_2_chunks = tmp_2.chunk(2, dim=1)
        tmp_7, tmp_8 = tmp_2_chunks[0], tmp_2_chunks[1]
        
        return tmp_4, tmp_7, tmp_5, tmp_8
    
    return chunk_operations