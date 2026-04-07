import torch
import triton
import triton.language as tl

def pattern(emb1, emb2, emb3, emb4, emb5, emb6, emb7, emb8):
    """
    Pattern for the sequential addition of embedding tensors.
    This matches the chain: tmp_35 = emb1 + emb2; tmp_36 = tmp_35 + emb3; ... tmp_42 = tmp_41 + emb8
    """
    tmp_35 = emb1 + emb2
    tmp_36 = tmp_35 + emb3
    tmp_37 = tmp_36 + emb4
    tmp_38 = tmp_37 + emb5
    tmp_39 = tmp_38 + emb6
    tmp_40 = tmp_39 + emb7
    tmp_41 = tmp_40 + emb8
    result = tmp_41
    return result

def replacement_args(emb1, emb2, emb3, emb4, emb5, emb6, emb7, emb8):
    # Return all embedding tensors as a tuple
    return (emb1, emb2, emb3, emb4, emb5, emb6, emb7, emb8)

@triton.jit
def parallel_embedding_sum_kernel(
    ptrs,
    output_ptr,
    n_elements,
    num_tensors,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for parallel addition of multiple embedding tensors.
    More efficient than sequential addition for GPU performance.
    """
    pid = tl.program_id(0)
    block_offset = pid * BLOCK_SIZE
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Initialize accumulator with zeros
    total_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Vectorized addition across all input tensors
    for tensor_idx in range(num_tensors):
        tensor_ptr = ptrs[tensor_idx]
        data = tl.load(tensor_ptr + offsets, mask=mask, other=0.0)
        total_sum = total_sum + data
    
    # Store the final result
    tl.store(output_ptr + offsets, total_sum, mask=mask)

@torch.fx.wrap
def parallel_embedding_sum_wrapper(tensors):
    """
    Wrapper function to launch optimized parallel embedding sum kernel.
    """
    if len(tensors) != 8:
        raise ValueError(f"Expected 8 tensors, got {len(tensors)}")
    
    # Get tensor properties from the first tensor
    n_elements = tensors[0].numel()
    dtype = tensors[0].dtype
    
    # Use optimal block size for embedding tensors (typically 1024 or 2048)
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype as inputs
    output = torch.empty_like(tensors[0])
    
    # Launch kernel with tensor pointers
    parallel_embedding_sum_kernel[(num_programs,)](
        ptrs=tensors,
        output_ptr=output,
        n_elements=n_elements,
        num_tensors=len(tensors),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """
    Returns the optimized parallel embedding sum function.
    """
    return parallel_embedding_sum_wrapper