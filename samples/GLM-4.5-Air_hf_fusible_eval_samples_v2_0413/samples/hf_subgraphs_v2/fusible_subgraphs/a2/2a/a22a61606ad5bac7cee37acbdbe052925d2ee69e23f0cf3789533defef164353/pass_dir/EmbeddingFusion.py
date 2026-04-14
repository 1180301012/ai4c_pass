import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_embedding_kernel(
    input_ids_ptr, token_type_ids_ptr, position_ids_ptr,
    word_weight_ptr, token_type_weight_ptr, position_weight_ptr,
    output_ptr,
    batch_size, seq_len, hidden_size,
    word_vocab_size, token_type_vocab_size, position_vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size * seq_len:
        return
    
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Load input indices
    input_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx)
    token_type_id = tl.load(token_type_ids_ptr + batch_idx * seq_len + seq_idx)
    position_id = tl.load(position_ids_ptr + batch_idx * seq_len + seq_idx)
    
    # Compute base pointer for output
    output_base = output_ptr + (batch_idx * seq_len + seq_idx) * hidden_size
    
    # Word embedding lookup
    word_emb_ptr = word_weight_ptr + input_id * hidden_size
    # Token type embedding lookup  
    token_type_emb_ptr = token_type_weight_ptr + token_type_id * hidden_size
    # Position embedding lookup
    position_emb_ptr = position_weight_ptr + position_id * hidden_size
    
    # Load embeddings and sum them element-wise
    for hid in range(0, hidden_size, BLOCK_SIZE):
        mask = hid + tl.arange(0, BLOCK_SIZE) < hidden_size
        
        word_val = tl.load(word_emb_ptr + hid, mask=mask, other=0.0)
        token_type_val = tl.load(token_type_emb_ptr + hid, mask=mask, other=0.0)
        position_val = tl.load(position_emb_ptr + hid, mask=mask, other=0.0)
        
        # Sum embeddings
        result = word_val + token_type_val + position_val
        
        # Store result
        tl.store(output_base + hid, result, mask=mask)

@torch.fx.wrap
def fused_embedding_lookup(input_ids, token_type_ids, position_ids, 
                          word_weight, token_type_weight, position_weight):
    batch_size, seq_len = input_ids.shape
    hidden_size = word_weight.shape[1]
    
    word_vocab_size = word_weight.shape[0]
    token_type_vocab_size = token_type_weight.shape[0] 
    position_vocab_size = position_weight.shape[0]
    
    output = torch.empty((batch_size, seq_len, hidden_size), 
                        dtype=word_weight.dtype, device=word_weight.device)
    
    BLOCK_SIZE = 64
    grid = (batch_size * seq_len,)
    
    fused_embedding_kernel[grid](
        input_ids, token_type_ids, position_ids,
        word_weight, token_type_weight, position_weight,
        output,
        batch_size, seq_len, hidden_size,
        word_vocab_size, token_type_vocab_size, position_vocab_size,
        BLOCK_SIZE
    )
    
    return output

def pattern(in_0, in_3, in_6, in_2, in_7, in_1):
    # First two embeddings are added
    tmp_7 = torch.nn.functional.embedding(in_0, in_3, 0, None, 2.0, False, False)
    tmp_8 = torch.nn.functional.embedding(in_6, in_2, None, None, 2.0, False, False)
    tmp_9 = tmp_7 + tmp_8
    
    # Third embedding is added to the result
    tmp_10 = torch.nn.functional.embedding(in_7, in_1, None, None, 2.0, False, False)
    result = tmp_9 + tmp_10
    
    return result

def replacement_args(in_0, in_3, in_6, in_2, in_7, in_1):
    return (in_0, in_3, in_6, in_2, in_7, in_1)

def replacement_func():
    return fused_embedding_lookup