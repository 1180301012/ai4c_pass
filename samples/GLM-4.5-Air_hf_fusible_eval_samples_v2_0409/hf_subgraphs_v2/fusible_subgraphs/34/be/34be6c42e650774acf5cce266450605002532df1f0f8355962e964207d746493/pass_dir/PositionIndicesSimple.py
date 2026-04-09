import torch
import triton
import triton.language as tl

# Pattern matching function - match embedding operation
def pattern(input_ids, weight_embedding):
    """
    Match embedding operation: torch.nn.functional.embedding(input_ids, weight_embedding, ...)
    """
    result = torch.nn.functional.embedding(input_ids, weight_embedding, 1, None, 2.0, False, False)
    return result

# Argument extraction function  
def replacement_args(input_ids, weight_embedding):
    return (input_ids, weight_embedding)

# Optimized embedding kernel
@triton.jit
def embedding_kernel(
    input_ids_ptr,
    weight_embedding_ptr,
    output_ptr,
    vocab_size,
    embed_dim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized embedding lookup using Triton"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input IDs (flatten the input)
    input_ids = tl.load(input_ids_ptr + offsets, mask=mask, other=0)
    
    # Calculate embedding weights offset: idx * embed_dim
    # We need to handle each embedding dimension separately
    seq_len = n_elements // embed_dim
    embed_idx = offsets % embed_dim  # Which dimension in the embedding
    seq_idx = offsets // embed_dim  # Which position in the sequence
    
    # Get the input_id for this sequence position
    input_id = tl.load(input_ids_ptr + seq_idx, mask=seq_idx < seq_len, other=0)
    
    # Calculate embedding weights offset: input_id * embed_dim + embed_idx
    weight_offset = input_id * embed_dim + embed_idx
    
    # Load embedding weights with bounds checking
    weight_mask = weight_offset < vocab_size * embed_dim
    embed_weight = tl.load(weight_embedding_ptr + weight_offset, mask=weight_mask, other=0.0)
    
    # Store output
    tl.store(output_ptr + offsets, embed_weight, mask=mask)

@torch.fx.wrap
def optimized_embedding(input_ids, weight_embedding):
    """Optimized embedding lookup using Triton"""
    input_shape = input_ids.shape
    embed_dim = weight_embedding.shape[1]
    n_elements = input_ids.numel()
    
    # Flatten input for easier processing
    input_ids_flat = input_ids.view(-1)
    
    # Prepare output tensor with correct shape: [batch_size, seq_len, embed_dim]
    total_output_elements = n_elements * embed_dim
    output = torch.empty(total_output_elements, dtype=weight_embedding.dtype, device=input_ids.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    embedding_kernel[(num_programs,)](
        input_ids_flat,
        weight_embedding,
        output,
        weight_embedding.shape[0],
        embed_dim,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output back to original input shape + embed_dim
    output = output.view(*input_shape, embed_dim)
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_embedding