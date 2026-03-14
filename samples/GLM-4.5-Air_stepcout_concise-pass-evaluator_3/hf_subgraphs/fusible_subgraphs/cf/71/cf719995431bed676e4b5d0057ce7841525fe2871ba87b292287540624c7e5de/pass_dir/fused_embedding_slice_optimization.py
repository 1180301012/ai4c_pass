import torch
import triton
import triton.language as tl
import math

# Pattern for embedding + slice operation (Type 1: skip first column)
def pattern(indices, weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    """Match embedding + slice pattern (skip first column)"""
    embeddings = torch.nn.functional.embedding(indices, weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    sliced_embeddings = embeddings[:, 1:]  # Skip first column
    return sliced_embeddings, embeddings

# Pattern for embedding + slice operation (Type 2: extract specific column)
def pattern_type2(indices, weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    """Match embedding + slice pattern (extract specific column)"""
    embeddings = torch.nn.functional.embedding(indices, weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    sliced_embeddings = embeddings[:, :, 1]  # Extract column 1
    return sliced_embeddings, embeddings

# Argument extraction functions
def replacement_args(indices, weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    return (indices, weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

# Pattern matching for Type 2
def replacement_args_type2(indices, weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    return (indices, weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

# Optimized Triton kernel for embedding + slice (Type 1: skip first column)
@triton.jit
def fused_embedding_slice_kernel_type1(
    indices_ptr,
    weights_ptr,
    embeddings_ptr,
    sliced_embeddings_ptr,
    num_sequences,
    sequence_length,
    vocab_size,
    embedding_dim,
    padding_idx: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused embedding kernel that skips first column"""
    # Each program handles one sequence position across all batches
    pid = tl.program_id(0)
    batch_id = pid // sequence_length
    seq_pos = pid % sequence_length
    
    if batch_id >= num_sequences:
        return
    
    # Load index for this position
    idx = tl.load(indices_ptr + batch_id * sequence_length + seq_pos)
    
    # Skip if padding index
    if idx == padding_idx:
        # Fill with zeros
        start_col = 1  # Skip first column
        for col in range(start_col, embedding_dim):
            offset = (batch_id * sequence_length + seq_pos) * embedding_dim + col
            tl.store(sliced_embeddings_ptr + offset, 0.0)
        
        # Also fill full embeddings for the return value
        for col in range(embedding_dim):
            offset = (batch_id * sequence_length + seq_pos) * embedding_dim + col
            tl.store(embeddings_ptr + offset, 0.0)
        return
    
    # Clamp index to valid range
    idx = tl.max(0, tl.min(idx, vocab_size - 1))
    
    # Compute base address for this embedding
    weight_offset = idx * embedding_dim
    
    # Load embedding vector (skip first column)
    start_col = 1  # Skip first column
    for col in range(start_col, embedding_dim):
        weight_idx = weight_offset + col
        embedding_val = tl.load(weights_ptr + weight_idx)
        
        # Store to sliced output
        output_offset = (batch_id * sequence_length + seq_pos) * embedding_dim + (col - start_col)
        tl.store(sliced_embeddings_ptr + output_offset, embedding_val)
    
    # Store full embeddings including first column for return value
    for col in range(embedding_dim):
        weight_idx = weight_offset + col
        embedding_val = tl.load(weights_ptr + weight_idx)
        output_offset = (batch_id * sequence_length + seq_pos) * embedding_dim + col
        tl.store(embeddings_ptr + output_offset, embedding_val)

# Optimized Triton kernel for embedding + slice (Type 2: extract specific column)
@triton.jit
def fused_embedding_slice_kernel_type2(
    indices_ptr,
    weights_ptr,
    embeddings_ptr,
    sliced_embeddings_ptr,
    num_sequences,
    sequence_length,
    vocab_size,
    embedding_dim,
    target_col: tl.constexpr,
    padding_idx: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused embedding kernel that extracts specific column"""
    # Each program handles one sequence position across all batches
    pid = tl.program_id(0)
    batch_id = pid // sequence_length
    seq_pos = pid % sequence_length
    
    if batch_id >= num_sequences:
        return
    
    # Load index for this position
    idx = tl.load(indices_ptr + batch_id * sequence_length + seq_pos)
    
    # Skip if padding index
    if idx == padding_idx:
        # Fill with zeros
        offset = (batch_id * sequence_length + seq_pos) * embedding_dim + target_col
        tl.store(sliced_embeddings_ptr + offset, 0.0)
        
        # Also fill full embeddings for the return value
        for col in range(embedding_dim):
            offset = (batch_id * sequence_length + seq_pos) * embedding_dim + col
            tl.store(embeddings_ptr + offset, 0.0)
        return
    
    # Clamp index to valid range
    idx = tl.max(0, tl.min(idx, vocab_size - 1))
    
    # Compute base address for this embedding
    weight_offset = idx * embedding_dim
    
    # Load specific column for sliced output
    embedding_val = tl.load(weights_ptr + weight_offset + target_col)
    output_offset = (batch_id * sequence_length + seq_pos) * embedding_dim + target_col
    tl.store(sliced_embeddings_ptr + output_offset, embedding_val)
    
    # Store full embeddings for return value
    for col in range(embedding_dim):
        weight_idx = weight_offset + col
        embedding_val = tl.load(weights_ptr + weight_idx)
        output_offset = (batch_id * sequence_length + seq_pos) * embedding_dim + col
        tl.store(embeddings_ptr + output_offset, embedding_val)

# Kernel wrapper for Type 1 (skip first column)
@torch.fx.wrap
def fused_embedding_slice_type1(indices, weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    num_sequences, sequence_length = indices.shape
    vocab_size, embedding_dim = weights.shape
    
    # Create output tensors
    full_embeddings = torch.empty((num_sequences, sequence_length, embedding_dim), dtype=weights.dtype, device=weights.device)
    sliced_embeddings = torch.empty((num_sequences, sequence_length, embedding_dim - 1), dtype=weights.dtype, device=weights.device)
    
    # Set up grid and launch kernel
    total_elements = num_sequences * sequence_length
    block_size = 512  # Optimized block size for GPU
    num_programs = (total_elements + block_size - 1) // block_size
    
    fused_embedding_slice_kernel_type1[(num_programs,)](
        indices_ptr=indices,
        weights_ptr=weights,
        embeddings_ptr=full_embeddings,
        sliced_embeddings_ptr=sliced_embeddings,
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx if padding_idx is not None else -1,
        BLOCK_SIZE=block_size,
    )
    
    return sliced_embeddings, full_embeddings

# Kernel wrapper for Type 2 (extract specific column)
@torch.fx.wrap
def fused_embedding_slice_type2(indices, weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    num_sequences, sequence_length = indices.shape
    vocab_size, embedding_dim = weights.shape
    
    # Create output tensors
    full_embeddings = torch.empty((num_sequences, sequence_length, embedding_dim), dtype=weights.dtype, device=weights.device)
    sliced_embeddings = torch.empty((num_sequences, sequence_length, embedding_dim), dtype=weights.dtype, device=weights.device)
    
    # Set up grid and launch kernel
    total_elements = num_sequences * sequence_length
    block_size = 512  # Optimized block size for GPU
    num_programs = (total_elements + block_size - 1) // block_size
    
    fused_embedding_slice_kernel_type2[(num_programs,)](
        indices_ptr=indices,
        weights_ptr=weights,
        embeddings_ptr=full_embeddings,
        sliced_embeddings_ptr=sliced_embeddings,
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        target_col=1,  # Extract column 1
        padding_idx=padding_idx if padding_idx is not None else -1,
        BLOCK_SIZE=block_size,
    )
    
    return sliced_embeddings, full_embeddings

# Replacement functions
def replacement_func_type1():
    return fused_embedding_slice_type1

def replacement_func_type2():
    return fused_embedding_slice_type2