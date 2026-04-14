import torch
import triton
import triton.language as tl

def pattern(input_0, weight_3, input_6, weight_2, input_7, weight_1):
    """Pattern to match three embedding operations followed by additions"""
    # First two embeddings are added
    tmp_7 = torch.nn.functional.embedding(input_0, weight_3, 0, None, 2.0, False, False)
    tmp_8 = torch.nn.functional.embedding(input_6, weight_2, None, None, 2.0, False, False)
    tmp_9 = tmp_7 + tmp_8
    
    # Third embedding is added to the result
    tmp_10 = torch.nn.functional.embedding(input_7, weight_1, None, None, 2.0, False, False)
    result = tmp_9 + tmp_10
    
    return result

def replacement_args(input_0, weight_3, input_6, weight_2, input_7, weight_1):
    hidden_size = weight_3.shape[1]
    return (input_0, weight_3, input_6, weight_2, input_7, weight_1, hidden_size)

def replacement_func():
    # Check if we need to return intermediate embeddings for the model output
    return multi_embedding_fused_kernel

@triton.jit
def fused_embedding_kernel(
    input_ids_ptr, token_type_ids_ptr, position_ids_ptr,
    word_weight_ptr, token_type_weight_ptr, position_weight_ptr,
    output_ptr,
    vocab_sizes, hidden_size,
    input_batch_size, seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for computing word + token_type + position embeddings"""
    # Get program ID
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    if batch_idx >= input_batch_size or seq_idx >= seq_len:
        return
    
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
def multi_embedding_fused_kernel(input_0, weight_3, input_6, weight_2, input_7, weight_1, hidden_size):
    # Get input shapes
    batch_size, seq_len = input_0.shape
    
    # Determine sizes of embedding tables
    word_vocab_size = weight_3.shape[0]
    token_type_vocab_size = weight_2.shape[0] 
    position_vocab_size = weight_1.shape[0]
    
    # Allocate output tensor
    output = torch.empty((batch_size, seq_len, hidden_size), dtype=weight_3.dtype, device=weight_3.device)
    
    # Set block size and launch grid
    BLOCK_SIZE = 64
    total_elements = batch_size * seq_len
    grid = (triton.cdiv(total_elements, 1),)
    
    # Launch fused kernel
    fused_embedding_kernel[grid](
        input_0, input_6, input_7,
        weight_3, weight_2, weight_1,
        output,
        (word_vocab_size, token_type_vocab_size, position_vocab_size),
        hidden_size,
        batch_size, seq_len,
        BLOCK_SIZE
    )
    
    return output