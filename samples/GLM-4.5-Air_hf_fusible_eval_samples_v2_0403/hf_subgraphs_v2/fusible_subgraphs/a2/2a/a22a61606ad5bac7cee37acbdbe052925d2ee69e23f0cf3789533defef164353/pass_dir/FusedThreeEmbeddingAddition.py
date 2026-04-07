import torch
import triton
import triton.language as tl

def pattern(in_0, in_3):
    # Simple pattern: just one embedding operation
    tmp_7 = torch.nn.functional.embedding(in_0, in_3, 0, None, 2.0, False, False)
    return tmp_7

def replacement_args(in_0, in_3):
    return (in_0, in_3)

@triton.jit
def single_embedding_kernel(
    input_ids_ptr, weight_ptr,
    output_ptr,
    num_tokens, hidden_size, vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID determines which token this program handles
    token_idx = tl.program_id(0)
    
    if token_idx >= num_tokens:
        return
    
    # Load input index for this token
    word_idx = tl.load(input_ids_ptr + token_idx)
    
    # Compute embedding index
    word_offset = word_idx * hidden_size
    word_embedding = tl.load(weight_ptr + word_offset, mask=(word_idx < vocab_size))
    
    # Store result
    output_offset = token_idx * hidden_size
    tl.store(output_ptr + output_offset, word_embedding)

@torch.fx.wrap
def optimized_embedding(input_ids, weight):
    # Get tensor properties
    batch_size, seq_len = input_ids.shape
    hidden_size = weight.shape[1]
    vocab_size = weight.shape[0]
    
    num_tokens = batch_size * seq_len
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, hidden_size), device=input_ids.device, dtype=weight.dtype)
    
    # Set block size and launch kernel
    BLOCK_SIZE = 256
    num_programs = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    single_embedding_kernel[(num_programs,)](
        input_ids_ptr=input_ids,
        weight_ptr=weight,
        output_ptr=output,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_embedding