import torch
import triton
import triton.language as tl

@triton.jit
def optimized_embedding_kernel(
    indices_ptr,
    weight_ptr,
    out_ptr,
    vocab_size,
    hidden_size,
    n_elements,
    block_size: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    
    # Load indices (already +2 offset applied)
    indices = tl.load(indices_ptr + offsets, mask=mask, dtype=tl.int64)
    
    # Handle padding: set indices to 0 for any index >= vocab_size to avoid out of bounds
    safe_indices = tl.minimum(indices, vocab_size - 1)
    
    # Calculate weight indices: each index maps to hidden_size consecutive elements
    weight_offsets = safe_indices * hidden_size + tl.arange(0, hidden_size)
    weight_offsets = weight_offsets.expand((block_size, hidden_size))
    
    # Load embeddings
    embeddings = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
    
    # Store result - reshape to [batch_size, seq_len, hidden_size] if needed
    # For simplicity, we store in flattened format and reshape in wrapper
    tl.store(out_ptr + offsets, embeddings, mask=mask)

@torch.fx.wrap
def optimized_embedding_forward(cache_positions, embed_weight):
    # Get tensor properties
    seq_len = cache_positions.shape[0]
    batch_size = 1  # Always 1 based on the input shapes
    vocab_size, hidden_size = embed_weight.shape
    
    # Apply offset + padding directly in the input preparation
    # This avoids the need for unsqueeze and separate addition
    indices = cache_positions + 2  # Add offset directly
    
    # Calculate output size and block size
    output_size = batch_size * seq_len * hidden_size
    block_size = 1024  # Optimal block size for most GPUs
    num_programs = (output_size + block_size - 1) // block_size
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, hidden_size), 
                       dtype=embed_weight.dtype, 
                       device=embed_weight.device)
    
    # Flatten indices for kernel
    flat_indices = indices.flatten()
    
    # Launch kernel with flattened output
    optimized_embedding_kernel[(num_programs,)](
        indices_ptr=flat_indices,
        weight_ptr=embed_weight,
        out_ptr=output.flatten(),
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_elements=output_size,
        block_size=block_size,
    )
    
    return output

# Pattern matching for embedding processing (optimizing unsqueeze + add + embedding)
def pattern(in_4, in_1):
    # Match: unsqueeze(0) + 2, then embedding
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    return tmp_11

def replacement_args(in_4, in_1):
    return (in_4, in_1)

def replacement_func():
    return optimized_embedding_forward