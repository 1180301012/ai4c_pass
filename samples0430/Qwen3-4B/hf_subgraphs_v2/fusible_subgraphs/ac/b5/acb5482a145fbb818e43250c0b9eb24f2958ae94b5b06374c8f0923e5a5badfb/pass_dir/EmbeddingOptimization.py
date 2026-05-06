import torch
import triton
import triton.language as tl

def pattern(input, weight):
    return torch.nn.functional.embedding(
        input,
        weight,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False
    )

def replacement_args(input, weight):
    return (input, weight)

@triton.jit
def embedding_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_n_elements,
    weight_n_elements,
    output_n_elements,
    BLOCK_SIZE,
):
    # Compute block index and range
    start = tl.program_id(0) * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, input_n_elements)
    
    # Load input indices
    input_ids = tl.load(input_ptr + start, tl.types.int32, tl.types.int32)
    
    # Process each element in the block
    for i in range(end - start):
        idx = input_ids[i]
        
        # Check if index is valid
        if idx < weight_n_elements:
            # Get embedding vector
            emb = tl.load(weight_ptr + idx * tl.int32(weight_n_elements))
            tl.store(output_ptr + start + i, emb)

@torch.fx.wrap
def embedding_wrapper(input, weight):
    # Allocate output with correct shape
    output = torch.empty(input.shape[0], input.shape[1], weight.shape[1], dtype=weight.dtype)
    
    # Launch kernel
    embedding_kernel[ ( (input.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE, ) ](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        input_n_elements=input.numel(),
        weight_n_elements=weight.numel(),
        output_n_elements=output.numel(),
        BLOCK_SIZE=256,
    )
    
    return output

def replacement_func():
    return embedding_wrapper