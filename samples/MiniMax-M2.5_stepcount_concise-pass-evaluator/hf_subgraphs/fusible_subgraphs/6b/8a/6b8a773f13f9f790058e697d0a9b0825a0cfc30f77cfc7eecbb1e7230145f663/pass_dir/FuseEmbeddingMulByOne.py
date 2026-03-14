import torch
import triton
import triton.language as tl


# Pattern matching function - matches embedding + multiply by 1.0 pattern
def pattern(in_0, in_1):
    """
    Match the pattern: embedding lookup followed by multiply by 1.0
    The multiply by 1.0 is a no-op and can be eliminated.
    
    torch.nn.functional.embedding parameters:
    - input: indices tensor  
    - weight: embedding table
    - padding_idx: index to ignore (1 in this case)
    - max_norm: maximum norm (None)
    - norm_type: norm type (2.0)
    - scale_grad_by_freq: whether to scale gradients by frequency (False)
    - sparse: whether to use sparse gradients (False)
    """
    # Embedding lookup with padding_idx=1, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False
    tmp_1 = torch.nn.functional.embedding(in_1, in_0, 1, None, 2.0, False, False)
    # Multiply by 1.0 - this is a no-op that can be eliminated
    tmp_2 = tmp_1 * 1.0
    return tmp_2


# Argument extraction function  
def replacement_args(in_0, in_1):
    """Extract the arguments needed for the replacement function."""
    return (in_0, in_1)


# Simple Triton kernel for embedding lookup - optimized for small inputs
@triton.jit
def embedding_kernel_simple(
    weight_ptr,  # embedding table [num_embeddings, embedding_dim]
    indices_ptr,  # input indices [batch_size, seq_len]
    output_ptr,  # output [batch_size, seq_len, embedding_dim]
    num_embeddings: tl.constexpr,
    embedding_dim: tl.constexpr,
    total_indices: tl.constexpr,
):
    """
    Simple Triton kernel for embedding lookup.
    Each program handles one index position and loads the full embedding.
    Uses vectorized load for efficiency.
    """
    # Each program handles one index position
    pid = tl.program_id(0)
    
    if pid >= total_indices:
        return
    
    # Load the index
    idx = tl.load(indices_ptr + pid)
    
    # Calculate base offset for this embedding in the weight table
    base_offset = idx * embedding_dim
    
    # Load the full embedding vector using vectorized load
    # Use multiple threads to load in parallel
    offsets = tl.arange(0, embedding_dim)
    mask = offsets < embedding_dim
    
    # Bounds check for the index
    if idx >= 0 and idx < num_embeddings:
        # Load the embedding vector - vectorized load
        weight_offsets = base_offset + offsets
        embeddings = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
    else:
        # Return zeros for out-of-bounds indices
        embeddings = tl.zeros((embedding_dim,), dtype=tl.float32)
    
    # Store the result
    output_offsets = pid * embedding_dim + offsets
    tl.store(output_ptr + output_offsets, embeddings, mask=mask)


@torch.fx.wrap
def triton_embedding_kernel(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the Triton embedding kernel.
    This eliminates the redundant multiply by 1.0 and uses an optimized kernel.
    """
    batch_size, seq_len = indices.shape
    total_indices = batch_size * seq_len
    num_embeddings, embedding_dim = weight.shape
    
    # Allocate output tensor
    output = torch.zeros(batch_size, seq_len, embedding_dim, 
                        dtype=weight.dtype, device=weight.device)
    
    # Launch kernel - one program per index position
    grid = (total_indices,)
    
    embedding_kernel_simple[grid](
        weight, 
        indices, 
        output,
        num_embeddings,
        embedding_dim,
        total_indices,
    )
    
    return output


# Replacement function - returns the Triton kernel wrapper
def replacement_func():
    return triton_embedding_kernel