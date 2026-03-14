import torch
import triton
import triton.language as tl

def pattern(input_ids, word_embeddings, position_ids, position_embeddings):
    # Two embedding operations with 2.0 scaling factor
    word_emb = torch.nn.functional.embedding(input_ids, word_embeddings, 1, None, 2.0, False, False)
    pos_emb = torch.nn.functional.embedding(position_ids, position_embeddings, 1, None, 2.0, False, False)
    
    # Just return the embeddings - the calling context will handle addition and scaling
    return word_emb, pos_emb

def replacement_args(input_ids, word_embeddings, position_ids, position_embeddings):
    return (input_ids, word_embeddings, position_ids, position_embeddings)

@triton.jit
def embedding_kernel(
    input_ids_ptr, embeddings_ptr, output_ptr,
    num_embeddings: tl.constexpr,
    embedding_dim: tl.constexpr,
    seq_len: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element
    program_id = tl.program_id(0)
    
    total_elements = batch_size * seq_len * embedding_dim
    if program_id >= total_elements:
        return
    
    # Calculate indices
    batch_id = program_id // (seq_len * embedding_dim)
    seq_id = (program_id % (seq_len * embedding_dim)) // embedding_dim
    emb_id = program_id % embedding_dim
    
    # Load input ID
    input_id = tl.load(input_ids_ptr + batch_id * seq_len + seq_id)
    
    # Calculate embedding offset with bounds checking
    if input_id < num_embeddings:
        offset = input_id * embedding_dim + emb_id
        embedding_val = tl.load(embeddings_ptr + offset)
        # Apply 2.0 scaling in the kernel
        scaled_val = embedding_val * 2.0
        tl.store(output_ptr + program_id, scaled_val)

@torch.fx.wrap
def optimized_embedding(input_ids, embeddings, position_ids, position_embeddings):
    seq_len = input_ids.size(-1)
    embedding_dim = embeddings.size(-1)
    batch_size = 1
    
    if input_ids.dim() == 2:
        batch_size, seq_len = input_ids.shape
    
    word_output = torch.zeros((batch_size, seq_len, embedding_dim), dtype=torch.float32, device=input_ids.device)
    pos_output = torch.zeros((batch_size, seq_len, embedding_dim), dtype=torch.float32, device=position_ids.device)
    
    # Process word embeddings
    if input_ids.dim() == 2:
        flat_input_ids = input_ids.view(-1)
        flat_word_output = word_output.view(-1)
        
        BLOCK_SIZE = 1024
        total_elements = batch_size * seq_len * embedding_dim
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        embedding_kernel[(num_programs,)](
            input_ids_ptr=flat_input_ids,
            embeddings_ptr=embeddings,
            output_ptr=flat_word_output,
            num_embeddings=embeddings.size(0),
            embedding_dim=embedding_dim,
            seq_len=seq_len,
            batch_size=batch_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Process position embeddings
        flat_position_ids = position_ids.view(-1)
        flat_pos_output = pos_output.view(-1)
        
        embedding_kernel[(num_programs,)](
            input_ids_ptr=flat_position_ids,
            embeddings_ptr=position_embeddings,
            output_ptr=flat_pos_output,
            num_embeddings=position_embeddings.size(0),
            embedding_dim=embedding_dim,
            seq_len=seq_len,
            batch_size=batch_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Handle unbatched case
        flat_input_ids = input_ids.view(-1)
        flat_word_output = word_output.view(-1)
        
        BLOCK_SIZE = 1024
        total_elements = seq_len * embedding_dim
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        embedding_kernel[(num_programs,)](
            input_ids_ptr=flat_input_ids,
            embeddings_ptr=embeddings,
            output_ptr=flat_word_output,
            num_embeddings=embeddings.size(0),
            embedding_dim=embedding_dim,
            seq_len=seq_len,
            batch_size=1,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        flat_position_ids = position_ids.view(-1)
        flat_pos_output = pos_output.view(-1)
        
        embedding_kernel[(num_programs,)](
            input_ids_ptr=flat_position_ids,
            embeddings_ptr=position_embeddings,
            output_ptr=flat_pos_output,
            num_embeddings=position_embeddings.size(0),
            embedding_dim=embedding_dim,
            seq_len=seq_len,
            batch_size=1,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return word_output, pos_output

def replacement_func():
    return optimized_embedding