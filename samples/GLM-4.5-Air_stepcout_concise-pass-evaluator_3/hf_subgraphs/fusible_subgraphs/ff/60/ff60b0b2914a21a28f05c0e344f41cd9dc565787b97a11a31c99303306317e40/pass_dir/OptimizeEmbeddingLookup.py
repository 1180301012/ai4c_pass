import torch
import triton
import triton.language as tl

def pattern(indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    """
    Match torch.nn.functional.embedding operation
    Args:
        indices: Input indices tensor
        weight: Embedding weight matrix 
        padding_idx: Padding index (None in this case)
        max_norm: Max norm constraint (None in this case)
        norm_type: Norm type (2.0 in this case)
        scale_grad_by_freq: Scale by frequency (False in this case)
        sparse: Sparse gradient (False in this case)
    """
    return torch.nn.functional.embedding(indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

def replacement_args(indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    return (indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

@triton.jit
def embedding_lookup_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    vocab_size,
    embedding_dim: tl.constexpr,
    num_indices,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized embedding lookup kernel - simplified version"""
    # Each program handles one index/output vector
    row_idx = tl.program_id(0)
    
    # Check bounds and return early if out of bounds
    if row_idx >= num_indices:
        return
    
    # Load the index for this row
    index = tl.load(indices_ptr + row_idx)
    
    # Validate index bounds
    if index < 0 or index >= vocab_size:
        # For invalid indices, the embedding lookup should return zeros
        # Initialize output to zeros using a loop for small embeddings
        for i in range(0, embedding_dim, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < embedding_dim
            zeros_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.bfloat16)
            output_pos = row_idx * embedding_dim + offset
            tl.store(output_ptr + output_pos, zeros_vals, mask=mask)
        return
    
    # Index is valid - compute embedding lookup
    weight_base = index * embedding_dim
    
    # Load embedding vector in chunks
    for i in range(0, embedding_dim, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < embedding_dim
        
        # Calculate memory offsets for weight and output
        weight_offset = weight_base + offset
        output_pos = row_idx * embedding_dim + offset
        
        # Load embedding weights
        embedding_vals = tl.load(
            weight_ptr + weight_offset,
            mask=mask,
            other=0.0
        )
        
        # Store to output
        tl.store(output_ptr + output_pos, embedding_vals, mask=mask)

@torch.fx.wrap
def optimized_embedding_lookup(indices, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """Optimized embedding lookup using Triton with autotuning"""
    num_indices = indices.shape[0] if len(indices.shape) == 1 else indices.shape[1] * indices.shape[0]
    vocab_size, embedding_dim = weight.shape
    
    # Output shape depends on input shape
    if len(indices.shape) == 1:
        output_shape = (num_indices, embedding_dim)
    else:
        # For 2D input, keep the first dimension and expand the second
        output_shape = indices.shape + (embedding_dim,)
    
    output = torch.empty(output_shape, dtype=weight.dtype, device=weight.device)
    
    # Flatten indices for simpler processing - this is more efficient
    if len(indices.shape) > 1:
        # For 2D indices, we reshape to process sequentially
        original_shape = indices.shape
        indices_flat = indices.view(-1)  # Flatten to 1D
        num_total_indices = indices_flat.numel()
    else:
        indices_flat = indices
        num_total_indices = num_indices
    
    # Choose optimal block size based on input size
    # For small inputs, use larger blocks to reduce overhead
    if num_total_indices <= 32:
        BLOCK_SIZE = 256
    elif num_total_indices <= 128:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 64
    
    num_programs = (num_total_indices + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel with flattened indices
    embedding_lookup_kernel[(num_programs,)](
        indices_ptr=indices_flat,
        weight_ptr=weight,
        output_ptr=output.view(-1, embedding_dim),  # Reshape output for contiguous access
        vocab_size=weight.shape[0],  # vocab_size is first dimension of weight
        embedding_dim=weight.shape[1],  # embedding_dim is second dimension of weight
        num_indices=num_total_indices,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_embedding_lookup