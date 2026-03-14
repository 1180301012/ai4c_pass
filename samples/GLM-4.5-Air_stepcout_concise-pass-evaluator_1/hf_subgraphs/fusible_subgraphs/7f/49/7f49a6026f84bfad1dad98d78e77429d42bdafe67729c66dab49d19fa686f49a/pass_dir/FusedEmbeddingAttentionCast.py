import torch
import triton
import triton.language as tl

def pattern(attn_mask, input_ids, pos_embeddings, word_embeddings, position_ids):
    # Embedding operations
    tmp_4 = torch.nn.functional.embedding(input_ids, word_embeddings, 1, None, 2.0, False, False)
    tmp_5 = torch.nn.functional.embedding(position_ids, pos_embeddings, 1, None, 2.0, False, False)
    
    # Add embeddings
    tmp_6 = tmp_4 + tmp_5
    
    # Apply attention mask
    tmp_7 = attn_mask.unsqueeze(-1)
    tmp_8 = tmp_6 * tmp_7
    
    # Type conversion
    tmp_9 = tmp_8.to(torch.float32)
    
    return tmp_9

def replacement_args(attn_mask, input_ids, pos_embeddings, word_embeddings, position_ids):
    return (attn_mask, input_ids, pos_embeddings, word_embeddings, position_ids)

@triton.jit
def fused_embedding_kernel(
    # Input pointers
    input_ids_ptr,  # First scalar input for efficient grid computation
    word_embeddings_ptr,
    pos_embeddings_ptr,
    position_ids_ptr,
    attn_mask_ptr,
    # Output pointer
    output_ptr,
    # Shape constants
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    embedding_dim: tl.constexpr,
    # Vocabulary sizes
    vocab_size: tl.constexpr,
    max_positions: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    # Flattened program index for better GPU occupancy
    pid = tl.program_id(0)
    total_programs = batch_size * seq_len
    linear_pid = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = linear_pid < total_programs
    
    # Convert linear index to batch and sequence coordinates
    seq_idx = linear_pid % seq_len
    batch_idx = linear_pid // seq_len
    
    # Load all scalar inputs for this position
    input_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx, mask=mask)
    position_id = tl.load(position_ids_ptr + batch_idx * seq_len + seq_idx, mask=mask)
    attn_mask_val = tl.load(attn_mask_ptr + batch_idx * seq_len + seq_idx, mask=mask)
    
    # Process embedding dimensions with vectorized loads
    embed_idx = tl.arange(0, BLOCK_SIZE)
    embed_mask = embed_idx < embedding_dim
    
    # Flatten the batch dimension for better memory access pattern
    flat_idx = batch_idx * seq_len + seq_idx
    
    # Word embedding lookup with improved locality
    word_offset = input_id * embedding_dim + embed_idx
    word_vecs = tl.load(word_embeddings_ptr + word_offset, mask=embed_mask)
    
    # Position embedding lookup
    pos_offset = position_id * embedding_dim + embed_idx
    pos_vecs = tl.load(pos_embeddings_ptr + pos_offset, mask=embed_mask)
    
    # Fused computation: add embeddings, apply mask, convert to float32
    result = (tl.cast(word_vecs, tl.float32) + tl.cast(pos_vecs, tl.float32)) * attn_mask_val
    
    # Optimized store pattern
    output_offset = flat_idx * embedding_dim + embed_idx
    tl.store(output_ptr + output_offset, result, mask=embed_mask)

@torch.fx.wrap
def fused_embedding_forward(attn_mask, input_ids, pos_embeddings, word_embeddings, position_ids):
    # Get input shapes
    batch_size, seq_len = attn_mask.shape
    _, embedding_dim = word_embeddings.shape
    
    # Optimized block size for better occupancy with these specific input sizes
    # Use smaller block size to avoid over-subscription for small inputs
    total_elements = batch_size * seq_len
    if total_elements <= 64:
        BLOCK_SIZE = 64   # Small inputs use smaller blocks
    elif total_elements <= 512:
        BLOCK_SIZE = 128  # Medium inputs
    else:
        BLOCK_SIZE = 256  # Larger inputs use larger blocks
    
    # Calculate flattened grid - one dimension for batch * sequence positions
    total_elements = batch_size * seq_len
    grid = ( (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    # Create output tensor with optimized layout
    output = torch.empty(batch_size, seq_len, embedding_dim, dtype=torch.float32, device=attn_mask.device)
    
    # Get vocabulary sizes for bounds checking
    vocab_size = word_embeddings.shape[0]
    max_positions = pos_embeddings.shape[0]
    
    # Launch highly optimized kernel with flattened addressing
    fused_embedding_kernel[grid](
        input_ids,      # Reordered for efficiency
        word_embeddings,
        pos_embeddings,
        position_ids,
        attn_mask,
        output,
        batch_size,
        seq_len,
        embedding_dim,
        vocab_size,
        max_positions,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_embedding_forward