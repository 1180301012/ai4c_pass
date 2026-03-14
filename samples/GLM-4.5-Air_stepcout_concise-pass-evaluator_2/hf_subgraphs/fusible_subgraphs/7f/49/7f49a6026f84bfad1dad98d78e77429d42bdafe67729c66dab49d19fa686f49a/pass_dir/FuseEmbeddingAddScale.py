import torch
import triton
import triton.language as tl

def pattern(input_ids, word_embeddings, position_ids, position_embeddings, attention_mask):
    # Two embedding operations
    word_emb = torch.nn.functional.embedding(input_ids, word_embeddings, 1, None, 2.0, False, False)
    pos_emb = torch.nn.functional.embedding(position_ids, position_embeddings, 1, None, 2.0, False, False)
    
    # Add embeddings
    combined_emb = word_emb + pos_emb
    
    # Scale by attention mask
    mask_scaled = combined_emb * attention_mask.unsqueeze(-1)
    
    return mask_scaled

def replacement_args(input_ids, word_embeddings, position_ids, position_embeddings, attention_mask):
    return (input_ids, word_embeddings, position_ids, position_embeddings, attention_mask)

@triton.jit
def fused_embedding_kernel(
    input_ids_ptr, word_emb_ptr, pos_ids_ptr, pos_emb_ptr, attention_mask_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    max_position: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch x sequence x hidden dimension
    program_id = tl.program_id(0)
    
    # Calculate batch, sequence, and hidden dimension indices
    total_elements = batch_size * seq_len * hidden_dim
    if program_id >= total_elements:
        return
    
    batch_id = program_id // (seq_len * hidden_dim)
    seq_id = (program_id % (seq_len * hidden_dim)) // hidden_dim
    hid_id = program_id % hidden_dim
    
    if batch_id >= batch_size or seq_id >= seq_len:
        return
    
    # Load input IDs and position IDs for this batch and sequence
    input_id = tl.load(input_ids_ptr + batch_id * seq_len + seq_id)
    pos_id = tl.load(pos_ids_ptr + batch_id * seq_len + seq_id)
    attention_val = tl.load(attention_mask_ptr + batch_id * seq_len + seq_id)
    
    # Calculate memory offsets for embedding lookup
    # Add bounds checking for safety
    if input_id < vocab_size and pos_id < max_position:
        word_emb_offset = input_id * hidden_dim + hid_id
        pos_emb_offset = pos_id * hidden_dim + hid_id
        
        # Load word and position embeddings with bounds checking
        word_emb_val = tl.load(word_emb_ptr + word_emb_offset)
        pos_emb_val = tl.load(pos_emb_ptr + pos_emb_offset)
        
        # Add embeddings, apply 2.0 scaling and attention mask scaling
        combined = (word_emb_val + pos_emb_val) * 2.0 * attention_val
        
        # Store result
        output_offset = batch_id * seq_len * hidden_dim + seq_id * hidden_dim + hid_id
        tl.store(output_ptr + output_offset, combined)

@torch.fx.wrap
def fused_embedding_add_scale(input_ids, word_embeddings, position_ids, position_embeddings, attention_mask):
    # Handle batch dimension
    if input_ids.dim() == 2:
        batch_size, seq_len = input_ids.shape
        hidden_dim = word_embeddings.size(-1)
        
        # Prepare output tensor with same shape as input but float32
        output_shape = (batch_size, seq_len, hidden_dim)
        output = torch.zeros(output_shape, dtype=torch.float32, device=input_ids.device)
        
        # Flatten input tensors for easier pointer access
        flat_input_ids = input_ids.view(-1)
        flat_position_ids = position_ids.view(-1)
        flat_attention_mask = attention_mask.view(-1)
        flat_output = output.view(-1)
        
        BLOCK_SIZE = 1024
        total_elements = batch_size * seq_len * hidden_dim
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel with proper grid
        fused_embedding_kernel[(num_programs,)](
            input_ids_ptr=flat_input_ids,
            word_emb_ptr=word_embeddings,
            pos_ids_ptr=flat_position_ids,
            pos_emb_ptr=position_embeddings,
            attention_mask_ptr=flat_attention_mask,
            output_ptr=flat_output,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            vocab_size=word_embeddings.size(0),
            max_position=position_embeddings.size(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # No batch dimension
        seq_len = input_ids.size(-1)
        hidden_dim = word_embeddings.size(-1)
        
        # Prepare output tensor
        output_shape = (seq_len, hidden_dim)
        output = torch.zeros(output_shape, dtype=torch.float32, device=input_ids.device)
        
        BLOCK_SIZE = 1024
        total_elements = seq_len * hidden_dim
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Treat as batch size 1
        fused_embedding_kernel[(num_programs,)](
            input_ids_ptr=input_ids,
            word_emb_ptr=word_embeddings,
            pos_ids_ptr=position_ids,
            pos_emb_ptr=position_embeddings,
            attention_mask_ptr=attention_mask,
            output_ptr=output,
            batch_size=1,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            vocab_size=word_embeddings.size(0),
            max_position=position_embeddings.size(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output

def replacement_func():
    return fused_embedding_add_scale