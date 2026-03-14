import torch
import triton
import triton.language as tl

def pattern(input_ids, word_embeddings, position_embeddings):
    # Optimized embedding lookup with better memory access patterns
    # This pattern matches the embedding operations but returns them pre-scaled
    word_emb = torch.nn.functional.embedding(input_ids, word_embeddings, 1, None, 2.0, False, False)
    pos_emb = torch.nn.functional.embedding(input_ids, position_embeddings, 1, None, 2.0, False, False)
    return word_emb, pos_emb

def replacement_args(input_ids, word_embeddings, position_embeddings):
    return (input_ids, word_embeddings, position_embeddings)

@triton.jit
def advanced_embedding_kernel(
    input_ids_ptr, emb1_ptr, emb2_ptr, 
    output1_ptr, output2_ptr,
    vocab_size1: tl.constexpr, vocab_size2: tl.constexpr,
    emb_dim: tl.constexpr, seq_len: tl.constexpr,
    batch_size: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * seq_len * emb_dim
    
    if pid >= total_elements:
        return
    
    # Calculate indices
    batch_id = pid // (seq_len * emb_dim)
    seq_id = (pid % (seq_len * emb_dim)) // emb_dim
    emb_id = pid % emb_dim
    
    # Load input ID once and reuse
    input_id = tl.load(input_ids_ptr + batch_id * seq_len + seq_id)
    
    # Process first embedding with bounds checking
    if input_id < vocab_size1:
        offset1 = input_id * emb_dim + emb_id
        emb1_val = tl.load(emb1_ptr + offset1)
        output1_val = emb1_val * 2.0
        tl.store(output1_ptr + pid, output1_val)
    else:
        tl.store(output1_ptr + pid, 0.0)
    
    # Process second embedding with bounds checking
    if input_id < vocab_size2:
        offset2 = input_id * emb_dim + emb_id
        emb2_val = tl.load(emb2_ptr + offset2)
        output2_val = emb2_val * 2.0
        tl.store(output2_ptr + pid, output2_val)
    else:
        tl.store(output2_ptr + pid, 0.0)

@torch.fx.wrap
def optimized_embedding_lookup(input_ids, word_embeddings, position_embeddings):
    seq_len = input_ids.size(-1)
    emb_dim = word_embeddings.size(-1)
    batch_size = 1
    
    if input_ids.dim() == 2:
        batch_size, seq_len = input_ids.shape
    
    # Output tensors
    word_output = torch.zeros((batch_size, seq_len, emb_dim), dtype=torch.float32, device=input_ids.device)
    pos_output = torch.zeros((batch_size, seq_len, emb_dim), dtype=torch.float32, device=input_ids.device)
    
    # Flatten tensors
    flat_input_ids = input_ids.view(-1)
    flat_word_output = word_output.view(-1)
    flat_pos_output = pos_output.view(-1)
    
    BLOCK_SIZE = 1024
    total_elements = batch_size * seq_len * emb_dim
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Optimized kernel that processes both embeddings in one pass per element
    advanced_embedding_kernel[(num_programs,)](
        input_ids_ptr=flat_input_ids,
        emb1_ptr=word_embeddings,
        emb2_ptr=position_embeddings,
        output1_ptr=flat_word_output,
        output2_ptr=flat_pos_output,
        vocab_size1=word_embeddings.size(0),
        vocab_size2=position_embeddings.size(0),
        emb_dim=emb_dim,
        seq_len=seq_len,
        batch_size=batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return word_output, pos_output

def replacement_func():
    return optimized_embedding_lookup