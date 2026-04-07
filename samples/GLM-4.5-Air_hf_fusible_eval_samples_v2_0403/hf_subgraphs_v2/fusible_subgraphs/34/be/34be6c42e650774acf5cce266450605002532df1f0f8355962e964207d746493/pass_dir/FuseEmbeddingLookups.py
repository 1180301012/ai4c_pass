import torch
import triton
import triton.language as tl

def word_embedding_pattern(input_tokens, word_embeddings, *args):
    """Match the word embedding operation"""
    result = torch.nn.functional.embedding(input_tokens, word_embeddings, 1, None, 2.0, False, False)
    return result

def replacement_args(input_tokens, word_embeddings, *args):
    return (input_tokens, word_embeddings)

# Removed unused fused embedding kernel code

@torch.fx.wrap
def optimized_embedding_lookup(input_tokens, word_embeddings, *args):
    """Optimized embedding lookup using Triton"""
    # Get input dimensions
    batch_size = input_tokens.shape[0]
    seq_length = input_tokens.shape[1]
    
    # Get embedding dimension from weight matrix
    embedding_dim = word_embeddings.shape[1]
    
    # Create output tensor
    output = torch.empty((batch_size, seq_length, embedding_dim), 
                        dtype=word_embeddings.dtype, 
                        device=word_embeddings.device)
    
    # Triton optimized kernel for embedding lookup
    @triton.jit
    def embedding_kernel(
        input_tokens_ptr, 
        weight_ptr, 
        output_ptr,
        num_embeddings: tl.constexpr,
        embedding_dim: tl.constexpr
    ):
        # Get block and program IDs
        batch_id = tl.program_id(0)
        seq_id = tl.program_id(1)
        embed_id = tl.program_id(2)
        
        # Calculate global offset
        offset = batch_id * seq_length * embedding_dim + seq_id * embedding_dim + embed_id
        
        # Calculate weight matrix offset
        token_id = tl.load(input_tokens_ptr + batch_id * seq_length + seq_id)
        weight_offset = token_id * embedding_dim + embed_id
        
        # Load embedding (with bounds check)
        mask = embed_id < embedding_dim
        embedding_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
        
        # Store result
        tl.store(output_ptr + offset, embedding_val, mask=mask)
    
    # Configure grid and launch kernel
    grid = (batch_size, seq_length, embedding_dim)
    embedding_kernel[grid](
        input_tokens,
        word_embeddings,
        output,
        word_embeddings.shape[0],
        embedding_dim
    )
    
    return output

def replacement_func():
    return optimized_embedding_lookup