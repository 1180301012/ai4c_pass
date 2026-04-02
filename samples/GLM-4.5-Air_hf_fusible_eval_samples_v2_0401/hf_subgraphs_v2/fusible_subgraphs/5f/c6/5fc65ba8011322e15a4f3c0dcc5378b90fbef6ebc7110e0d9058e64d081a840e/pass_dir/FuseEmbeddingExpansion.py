import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    return tmp_2

@triton.jit
def embedding_expansion_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    seq_len,
    vocab_size,
    embed_dim,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_EMBED: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_embed = tl.program_id(1)
    
    # Global offsets
    batch_offset = pid_batch * BLOCK_BATCH
    embed_offset = pid_embed * BLOCK_EMBED
    
    # Create masks for batch and embedding dimensions
    batch_mask = batch_offset + tl.arange(0, BLOCK_BATCH) < batch_size
    embed_mask = embed_offset + tl.arange(0, BLOCK_EMBED) < embed_dim
    
    # Process sequence positions
    for seq_pos in range(0, seq_len + 2):
        # Calculate output position offset
        output_base = (batch_offset * (seq_len + 2) + seq_pos) * (embed_dim * 3) + embed_offset
        
        # Determine which part of the output we're writing to
        if embed_offset < embed_dim:
            # Section 1: tmp_4 (tokens 1..seq_len, padded with 1 zero at start)
            if seq_pos == 0:
                # Position 0: all zeros (padded start)
                for i in range(0, min(BLOCK_EMBED, embed_dim - embed_offset)):
                    output_idx = output_base + i
                    if batch_mask[0] and embed_mask[i]:
                        tl.store(output_ptr + output_idx, 0.0)
            
            # Section 2: original embeddings (tmp_2)
            elif 1 <= seq_pos <= seq_len:
                # Load the embedding for token at position (seq_pos - 1)
                token_idx = seq_pos - 1
                for i in range(0, min(BLOCK_EMBED, embed_dim - embed_offset)):
                    output_idx = output_base + embed_dim + i  # Middle section
                    emb_ptr = weight_ptr + token_idx * vocab_size * embed_dim + embed_offset + i
                    
                    if batch_mask[0] and token_idx < vocab_size and embed_mask[i]:
                        embedding_val = tl.load(emb_ptr)
                        tl.store(output_ptr + output_idx, embedding_val)
                    else:
                        tl.store(output_ptr + output_idx, 0.0)
            
            # Section 3: tmp_6 (tokens 0..seq_len-1, padded with 1 zero at end)
            elif seq_pos == seq_len + 1:
                for i in range(0, min(BLOCK_EMBED, embed_dim - embed_offset)):
                    output_idx = output_base + embed_dim * 2 + i  # Last section
                    if batch_mask[0] and embed_mask[i]:
                        tl.store(output_ptr + output_idx, 0.0)

@torch.fx.wrap
def optimized_embedding(input_ids, weight):
    # Simple approach: use one program per batch element and token combination
    batch_size, seq_len = input_ids.shape
    vocab_size, embed_dim = weight.shape
    
    output = torch.empty((batch_size, seq_len, embed_dim), dtype=weight.dtype, device=weight.device)
    
    @triton.jit
    def simple_embedding_kernel(
        input_ids_ptr,
        weight_ptr, 
        output_ptr,
        batch_size,
        seq_len,
        vocab_size,
        embed_dim,
    ):
        # Get program IDs - each program handles one batch element and one sequence position
        batch_idx = tl.program_id(0)
        seq_idx = tl.program_id(1)
        
        # Check bounds
        if batch_idx >= batch_size or seq_idx >= seq_len:
            return
            
        # Load token ID for this position
        token_idx = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx)
        
        # Check if token index is valid
        if token_idx >= vocab_size:
            return
            
        # Calculate output pointer for this embedding
        emb_base = weight_ptr + token_idx * embed_dim
        out_base = output_ptr + batch_idx * seq_len * embed_dim + seq_idx * embed_dim
        
        # Load and store embedding values
        for j in range(0, embed_dim):
            emb_val = tl.load(emb_base + j)
            tl.store(out_base + j, emb_val)
    
    # Calculate grid dimensions - one program per batch per sequence position
    num_batches = batch_size
    num_seqs = seq_len
    
    # Launch kernel - one program per batch element and sequence position
    simple_embedding_kernel[(num_batches, num_seqs)](
        input_ids_ptr=input_ids,
        weight_ptr=weight,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
    )
    
    return output

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    return optimized_embedding