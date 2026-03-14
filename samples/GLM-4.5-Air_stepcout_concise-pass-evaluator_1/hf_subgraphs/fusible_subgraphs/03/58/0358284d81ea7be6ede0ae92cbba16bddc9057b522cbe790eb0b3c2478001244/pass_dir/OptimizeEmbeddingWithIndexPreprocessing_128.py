import torch
from torch import device
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    """
    Match the embedding operation with index preprocessing for seq_len=128
    """
    tmp_1 = torch.arange(128, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_2 = tmp_1.view(1, -1)
    tmp_3 = in_1 - tmp_2
    tmp_4 = tmp_3 + 2048
    tmp_5 = tmp_4 - 1
    tmp_6 = torch.nn.functional.embedding(tmp_5, in_0, None, None, 2.0, False, False)
    tmp_7 = tmp_6.to(dtype=torch.float32)
    return tmp_7

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_embedding_kernel(
    weight_ptr,               # embedding weights: [num_embeddings, embedding_dim]
    indices_ptr,             # input indices: [batch_size, seq_len] 
    out_ptr,                 # output: [batch_size, seq_len, embedding_dim]
    batch_size,              # batch dimension
    seq_len,                 # sequence length
    num_embeddings,          # vocabulary size
    embedding_dim,           # embedding dimension
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
):
    # Generate global position for this program
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Create range tensor on the fly and compute final indices
    # Original: in_1 - arange(seq_len) + 2048 - 1
    #           = in_1 - arange(seq_len) + 2047
    
    # Start position in indices tensor
    idx_offset = batch_idx * seq_len + seq_idx
    
    # Load the base index and compute the transformed index
    base_idx = tl.load(indices_ptr + idx_offset)
    range_val = tl.arange(0, BLOCK_SIZE_SEQ) + seq_idx * BLOCK_SIZE_SEQ
    transformed_idx = base_idx - range_val + 2047
    
    # Handle bounds checking
    mask = (transformed_idx >= 0) & (transformed_idx < num_embeddings)
    
    # Load embedding weight rows
    weight_rows = tl.load(weight_ptr + transformed_idx * embedding_dim, mask=mask, other=0.0)
    
    # Store result
    out_offset = (batch_idx * seq_len + seq_idx) * embedding_dim + tl.arange(0, BLOCK_SIZE_SEQ)
    tl.store(out_ptr + out_offset, weight_rows, mask=tl.arange(0, BLOCK_SIZE_SEQ) < BLOCK_SIZE_SEQ)

def triton_embedding_optimized(weight, indices, batch_size, num_embeddings, embedding_dim):
    """Wrapper for optimized embedding kernel for seq_len=128"""
    seq_len = 128  # Fixed for this pass
    
    BLOCK_SIZE_SEQ = 128
    BLOCK_SIZE_BATCH = 32
    
    grid = (
        (batch_size + BLOCK_SIZE_BATCH - 1) // BLOCK_SIZE_BATCH,
        (seq_len + BLOCK_SIZE_SEQ - 1) // BLOCK_SIZE_SEQ,
    )
    
    # Create output tensor
    out_shape = (batch_size, seq_len, embedding_dim)
    output = torch.empty(out_shape, dtype=torch.float32, device=weight.device)
    
    # Launch kernel
    optimized_embedding_kernel[grid](
        weight_ptr=weight,
        indices_ptr=indices,
        out_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
    )
    
    return output

@torch.fx.wrap
def optimized_embedding_forward_128(weight, indices):
    """Optimized embedding function for seq_len=128 with fused index preprocessing"""
    batch_size, seq_len = indices.shape
    num_embeddings, embedding_dim = weight.shape
    
    # Use optimized Triton kernel for sequence length 128
    return triton_embedding_optimized(weight, indices, batch_size, num_embeddings, embedding_dim)

def replacement_func():
    return optimized_embedding_forward_128