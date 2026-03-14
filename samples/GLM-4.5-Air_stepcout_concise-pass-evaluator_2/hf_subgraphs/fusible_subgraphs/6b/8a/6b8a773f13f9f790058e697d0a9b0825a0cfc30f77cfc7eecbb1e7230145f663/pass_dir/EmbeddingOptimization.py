import torch
import triton
import triton.language as tl

def pattern(input_ids, weight_tensor):
    # Match the full computation: embedding followed by multiplication by 1.0
    tmp_1 = torch.nn.functional.embedding(input_ids, weight_tensor, 1, None, 2.0, False, False)
    tmp_2 = tmp_1 * 1.0
    return tmp_2

def replacement_args(input_ids, weight_tensor):
    return (input_ids, weight_tensor)

@triton.jit
def optimized_embedding_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    padding_idx,
    weight_stride0,
    embedding_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized embedding kernel that eliminates redundant operations"""
    pid = tl.program_id(0)
    
    # Load the input index
    input_idx = tl.load(input_ids_ptr + pid)
    
    # Handle padding index - return zero vector if this is the padding index
    if input_idx == padding_idx:
        tl.store(output_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), 
                 0.0, mask=tl.arange(0, BLOCK_SIZE) < embedding_dim)
        return
    
    # Calculate pointer to the embedding vector
    embedding_ptr = weight_ptr + input_idx * weight_stride0
    
    # Load the embedding vector directly - no multiplication needed
    tl.store(output_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), 
             tl.load(embedding_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < embedding_dim, other=0.0), 
             mask=tl.arange(0, BLOCK_SIZE) < embedding_dim)

@torch.fx.wrap
def optimized_embedding_lookup(input_ids, weight_tensor):
    """
    Optimized embedding lookup that:
    1. Handles device transfer efficiently
    2. Eliminates redundant multiplication by 1.0
    3. Uses efficient Triton kernel for embedding lookup
    """
    # Handle device transfer efficiently (only transfer if needed)
    if weight_tensor.device != input_ids.device:
        # For small weight tensors, consider caching on GPU
        # But for now, just transfer as needed
        weight_tensor = weight_tensor.to(input_ids.device)
    
    # Ensure contiguous memory layout
    input_ids = input_ids.contiguous()
    weight_tensor = weight_tensor.contiguous()
    
    input_size = input_ids.numel()
    embedding_dim = weight_tensor.shape[1]
    
    # Create output tensor
    output = torch.empty((input_size, embedding_dim), dtype=weight_tensor.dtype, device=input_ids.device)
    
    if input_size > 0:
        # Use optimized kernel for small batch sizes
        BLOCK_SIZE = embedding_dim
        grid = (input_size,)
        
        optimized_embedding_kernel[grid](
            input_ids,
            weight_tensor,
            output,
            1,  # padding_idx
            weight_tensor.stride(0),
            embedding_dim,
            BLOCK_SIZE,
        )
    
    return output

def replacement_func():
    return optimized_embedding_lookup