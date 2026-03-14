import torch
import triton
import triton.language as tl


def pattern(input_ids, weight):
    """
    Match the pattern: embedding lookup
    """
    # Match exact signature from model.py - just the embedding operation
    result = torch.nn.functional.embedding(input_ids, weight, 0, None, 2.0, False, False)
    return result


def replacement_args(input_ids, weight):
    return (input_ids, weight)


@triton.jit
def embedding_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    seq_len,
    embedding_dim,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized embedding lookup kernel
    Each program handles one position in the batch x seq_len grid
    """
    # Get position in batch and sequence
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Load the token id
    token_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx)
    
    # Calculate offset in weight matrix for this token
    weight_offset = token_id * embedding_dim
    
    # Calculate output offset
    output_offset = (batch_idx * seq_len + seq_idx) * embedding_dim
    
    # Load and store embedding in blocks
    for block_start in range(0, embedding_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < embedding_dim
        
        # Load embedding values
        embedding_vals = tl.load(weight_ptr + weight_offset + offsets, mask=mask, other=0.0)
        
        # Store to output
        tl.store(output_ptr + output_offset + offsets, embedding_vals, mask=mask)


@torch.fx.wrap
def optimized_embedding(input_ids, weight):
    """
    Optimized embedding lookup using Triton
    """
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    embedding_dim = weight.shape[1]
    
    # Allocate output for full embedding
    output = torch.empty(
        (batch_size, seq_len, embedding_dim),
        dtype=weight.dtype,
        device=weight.device
    )
    
    # Launch kernel for full embedding
    BLOCK_SIZE = 128
    grid = (batch_size * seq_len,)
    
    embedding_kernel[grid](
        input_ids,
        weight,
        output,
        batch_size,
        seq_len,
        embedding_dim,
        weight.shape[0],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return optimized_embedding