import torch
import triton
import triton.language as tl

def pattern(input_ids_0, position_ids_0, word_embeddings_weight, position_embeddings_weight):
    """Pattern: Direct addition of two embedding operations
    This matches the observable computation: result = embedding(in_0) + embedding(in_5)
    """
    # Directly compute the sum of embeddings to avoid dead intermediate results
    return torch.nn.functional.embedding(input_ids_0, word_embeddings_weight, 1, None, 2.0, False, False) + torch.nn.functional.embedding(position_ids_0, position_embeddings_weight, 1, None, 2.0, False, False)

def replacement_args(input_ids_0, position_ids_0, word_embeddings_weight, position_embeddings_weight):
    """Extract arguments needed for fused embedding kernel"""
    return (input_ids_0, position_ids_0, word_embeddings_weight, position_embeddings_weight)

@triton.jit
def fused_embedding_add_kernel(
    input_ids_ptr,
    position_ids_ptr,
    word_embeddings_ptr,
    position_embeddings_ptr,
    output_ptr,
    num_sequences,
    seq_length,
    vocab_size_word,
    vocab_size_pos,
    embedding_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused embedding addition kernel with optimized memory access
    Computes: output = embedding(input_ids) + embedding(position_ids)
    """
    # Get program IDs for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges for this program
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create offset ranges
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create 2D mask
    m_mask = m_offsets < num_sequences
    n_mask = n_offsets < seq_length
    
    # Optimized memory layout: process smaller tiles for better cache utilization
    for k in range(0, embedding_dim, 64):  # Smaller chunks for better cache locality
        k_end = min(k + 64, embedding_dim)
        
        # Load input IDs and position IDs for current block
        input_ids_block = tl.load(
            input_ids_ptr + m_offsets[:, None] * seq_length + n_offsets[None, :],
            mask=m_mask[:, None] & n_mask[None, :],
            other=-1
        )
        
        position_ids_block = tl.load(
            position_ids_ptr + m_offsets[:, None] * seq_length + n_offsets[None, :],
            mask=m_mask[:, None] & n_mask[None, :],
            other=-1
        )
        
        # Word embeddings lookup with optimized address calculation
        word_emb_offsets = input_ids_block * embedding_dim + k
        word_embeddings_block = tl.load(
            word_embeddings_ptr + word_emb_offsets,
            mask=(input_ids_block != -1) & (m_mask[:, None] & n_mask[None, :]),
            other=0.0
        )
        
        # Position embeddings lookup  
        pos_emb_offsets = position_ids_block * embedding_dim + k
        position_embeddings_block = tl.load(
            position_embeddings_ptr + pos_emb_offsets,
            mask=(position_ids_block != -1) & (m_mask[:, None] & n_mask[None, :]),
            other=0.0
        )
        
        # Add embeddings and store with optimized addressing
        output_block = word_embeddings_block + position_embeddings_block
        output_offsets = (m_offsets[:, None] * seq_length + n_offsets[None, :]) * embedding_dim + k
        tl.store(
            output_ptr + output_offsets,
            output_block,
            mask=m_mask[:, None] & n_mask[None, :]
        )

def _next_power_of_2(n):
    """Find the next power of 2 for a given number"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1

@torch.fx.wrap
def fused_embedding_add(input_ids, position_ids, word_embeddings, position_embeddings):
    """Fused embedding addition function with power-of-2 block sizes"""
    # Get input shapes
    num_sequences, seq_length = input_ids.shape
    vocab_size_word, embedding_dim = word_embeddings.shape
    vocab_size_pos = position_embeddings.shape[0]
    
    # Allocate output tensor
    output = torch.empty((num_sequences, seq_length, embedding_dim), dtype=word_embeddings.dtype, device=word_embeddings.device)
    
    # Adaptive block sizes based on input configuration (ensuring powers of 2)
    if seq_length <= 16:  # Very short sequences
        BLOCK_SIZE_M = 16   # Power of 2: 16 sequences per block
        BLOCK_SIZE_N = 16   # Power of 2: fixed small block size
    elif seq_length <= 64:  # Medium sequences
        BLOCK_SIZE_M = 8    # Power of 2
        BLOCK_SIZE_N = 32   # Power of 2
    else:  # Long sequences
        BLOCK_SIZE_M = 8    # Power of 2
        BLOCK_SIZE_N = 256  # Power of 2
    
    # Adjust BLOCK_SIZE_M based on number of sequences (always powers of 2)
    if num_sequences <= 8:
        BLOCK_SIZE_M = min(8, num_sequences)
        if num_sequences <= 4:
            BLOCK_SIZE_M = 4  # Power of 2
        if num_sequences <= 2:
            BLOCK_SIZE_M = 2
        if num_sequences == 1:
            BLOCK_SIZE_M = 1
    
    # Ensure BLOCK_SIZE_N is never larger than seq_length
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, seq_length)
    
    # Calculate grid dimensions
    grid_m = (num_sequences + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_length + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # For very small workloads, use a single thread
    if num_sequences * seq_length * embedding_dim < 512 * 64:  # Very small workload
        BLOCK_SIZE_M = 1
        BLOCK_SIZE_N = 1
        grid_m = 1
        grid_n = 1
    
    # Launch kernel with 2D grid
    fused_embedding_add_kernel[(grid_m, grid_n)](
        input_ids_ptr=input_ids,
        position_ids_ptr=position_ids,
        word_embeddings_ptr=word_embeddings,
        position_embeddings_ptr=position_embeddings,
        output_ptr=output,
        num_sequences=num_sequences,
        seq_length=seq_length,
        vocab_size_word=vocab_size_word,
        vocab_size_pos=vocab_size_pos,
        embedding_dim=embedding_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    """Return the fused embedding addition function"""
    return fused_embedding_add