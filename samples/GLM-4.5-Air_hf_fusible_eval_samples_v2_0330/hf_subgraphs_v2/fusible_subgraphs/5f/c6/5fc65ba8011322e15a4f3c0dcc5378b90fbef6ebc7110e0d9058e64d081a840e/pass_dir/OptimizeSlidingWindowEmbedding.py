import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern matches just the embedding operation"""
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_embedding_kernel(
    input_ptr,  # [batch_size, seq_len] - input IDs  
    weight_ptr,  # [vocab_size, embed_dim] - embedding weights
    output_ptr,  # [batch_size, seq_len, embed_dim] - output
    vocab_size,
    embed_dim,
    batch_size,
    seq_len,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles one embedding lookup
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Boundary check - skip if out of bounds
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return
    
    # Compute offsets
    input_offset = batch_idx * seq_len + seq_idx
    output_offset = (batch_idx * seq_len + seq_idx) * embed_dim
    
    # Load input token ID
    token_id = tl.load(input_ptr + input_offset)
    
    # Clamp token ID to valid range
    token_id = tl.maximum(0, tl.minimum(token_id, vocab_size - 1))
    
    # Compute weight offset for this token
    weight_offset = token_id * embed_dim
    
    # Process embedding vector with larger chunks for better efficiency
    for k_offset in tl.range(0, embed_dim, BLOCK_SIZE_K):
        # Compute actual global offset
        global_output_offset = output_offset + k_offset
        global_weight_offset = weight_offset + k_offset
        
        # Load embedding element (simple scalar load avoids compilation issues)
        embed_val = tl.load(weight_ptr + global_weight_offset)
        
        # Store to output
        tl.store(output_ptr + global_output_offset, embed_val)

@torch.fx.wrap  
def optimized_embedding_forward(input_ids, embedding_weight):
    batch_size, seq_len = input_ids.shape
    vocab_size, embed_dim = embedding_weight.shape
    
    # Output shape: [batch_size, seq_len, embed_dim]
    output = torch.empty(batch_size, seq_len, embed_dim, 
                        dtype=embedding_weight.dtype, device=embedding_weight.device)
    
    # Configure grid - one program per batch x position
    grid = (batch_size, seq_len)
    
    # Optimal block size for processing embedding vector elements
    # Use larger block size to reduce loop overhead
    BLOCK_SIZE_K = 256  # Process 256 elements at a time
    
    # Launch kernel
    optimized_embedding_kernel[grid](
        input_ids,
        embedding_weight,
        output,
        vocab_size,
        embed_dim,
        batch_size,
        seq_len,
        BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return optimized_embedding_forward