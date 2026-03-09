import torch
import triton
import triton.language as tl

def pattern(input_ids, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    # Match the embedding operation from the target computation
    return torch.nn.functional.embedding(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

def replacement_args(input_ids, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    # Extract arguments needed for the replacement
    return (input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

@triton.jit
def embedding_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    vocab_size,
    embedding_dim: tl.constexpr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for embedding lookup."""
    pid = tl.program_id(0)
    
    # Calculate the starting index for this program
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return
    
    # Load the input ID (index)
    input_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx)
    
    # Use power of 2 for arange (2048 is the next power of 2 after 1536)
    range_size = 2048
    
    # Calculate the embedding offset
    base_offset = input_id * embedding_dim
    offset = base_offset + tl.arange(0, range_size)
    
    # Load the corresponding embedding vector with bounds checking
    mask = (offset < (vocab_size * embedding_dim)) & (tl.arange(0, range_size) < embedding_dim)
    embedding_vector = tl.load(weight_ptr + offset, mask=mask, other=0.0)
    
    # Calculate the output offset
    output_offset = pid * embedding_dim + tl.arange(0, range_size)
    
    # Store the result
    tl.store(output_ptr + output_offset, embedding_vector, mask=mask)

@torch.fx.wrap
def optimized_embedding(input_ids, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """Optimized embedding lookup using Triton kernel."""
    batch_size, seq_len = input_ids.shape
    vocab_size, embedding_dim = weight.shape
    
    # Use power of 2 for arange (2048 is the next power of 2 after 1536)
    range_size = 2048
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, embedding_dim), dtype=weight.dtype, device=weight.device)
    
    # Determine block size and grid
    BLOCK_SIZE = 1024
    grid_size = batch_size * seq_len
    
    # Launch the kernel
    embedding_kernel[grid_size](
        input_ids,
        weight,
        output,
        vocab_size,
        embedding_dim,
        batch_size,
        seq_len,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """Return the optimized embedding function."""
    return optimized_embedding