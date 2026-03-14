import torch
import triton
import triton.language as tl

# Pattern: optimize embedding lookup
def pattern(in_5, in_3):
    """
    Simple pattern: just the embedding lookup
    """
    tmp_5 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    return tmp_5

def replacement_args(in_5, in_3):
    return (in_5, in_3)

@triton.jit
def optimized_embedding_kernel(
    indices_ptr, weight_ptr, output_ptr,
    batch_size, seq_len, hidden_dim,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized embedding lookup kernel with vectorized access
    """
    # Each program handles one position in the batch*seq_len dimension
    pid = tl.program_id(0)
    
    # Load embedding index for this position
    embedding_idx = tl.load(indices_ptr + pid)
    
    # Calculate output base offset
    output_offset = pid * hidden_dim
    weight_offset = embedding_idx * hidden_dim
    
    # Process in blocks of BLOCK_SIZE - vectorized loads
    num_blocks = tl.cdiv(hidden_dim, BLOCK_SIZE)
    for i in range(num_blocks):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        
        # Vectorized load from weight matrix
        vals = tl.load(weight_ptr + weight_offset + offsets, mask=mask, other=0.0)
        
        # Vectorized store to output
        tl.store(output_ptr + output_offset + offsets, vals, mask=mask)

@torch.fx.wrap
def optimized_embedding(in_5, in_3):
    """
    Optimized embedding lookup using Triton
    """
    batch_size, seq_len = in_5.shape
    vocab_size, hidden_dim = in_3.shape
    
    output = torch.empty(batch_size, seq_len, hidden_dim, dtype=in_3.dtype, device=in_3.device)
    
    num_positions = batch_size * seq_len
    BLOCK_SIZE = 256  # Optimal for larger hidden dims
    
    grid = (num_positions,)
    
    optimized_embedding_kernel[grid](
        in_5, in_3, output,
        batch_size, seq_len, hidden_dim,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_embedding