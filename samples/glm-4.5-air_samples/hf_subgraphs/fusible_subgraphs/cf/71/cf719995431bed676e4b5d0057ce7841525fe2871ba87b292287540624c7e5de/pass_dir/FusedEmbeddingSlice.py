import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
    ],
    key=['batch_size', 'seq_len', 'hidden_dim'],
)
@triton.jit
def embedding_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Triton kernel for embedding lookup."""
    row_offset = tl.program_id(0) * BLOCK_SIZE_M
    col_offset = tl.program_id(1) * BLOCK_SIZE_N
    
    row_offsets = row_offset + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
    
    # Mask for out-of-bounds
    row_mask = row_offsets < batch_size * seq_len
    col_mask = col_offsets < hidden_dim
    mask = row_mask & col_mask
    
    # Calculate batch and sequence indices
    batch_idx = row_offsets // seq_len
    seq_idx = row_offsets % seq_len
    
    # Calculate flat index for input_ids
    input_ids_flat_idx = batch_idx * seq_len + seq_idx
    input_ids_vals = tl.load(input_ids_ptr + input_ids_flat_idx, mask=row_mask, other=0)
    
    # Clamp indices to valid vocabulary range
    input_ids_vals = tl.where(input_ids_vals < vocab_size, input_ids_vals, 0)
    input_ids_vals = tl.where(input_ids_vals >= 0, input_ids_vals, 0)
    
    # Calculate embedding offsets (vocab_idx * hidden_dim + hidden_idx)
    embedding_offsets = input_ids_vals * hidden_dim + col_offsets
    embedding_vals = tl.load(embedding_weight_ptr + embedding_offsets, mask=mask, other=0.0)
    
    # Calculate output position
    output_flat_idx = row_offsets * hidden_dim + col_offsets
    tl.store(output_ptr + output_flat_idx, embedding_vals, mask=mask)


@torch.fx.wrap
def triton_embedding(input_ids, embedding_weight):
    """Triton kernel for embedding lookup."""
    batch_size, seq_len = input_ids.shape
    hidden_dim = embedding_weight.shape[1]
    vocab_size = embedding_weight.shape[0]
    
    # Allocate output
    output = torch.empty((batch_size, seq_len, hidden_dim), 
                       device=input_ids.device, 
                       dtype=embedding_weight.dtype)
    
    # Configure block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    
    # Calculate grid
    num_m = (batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n = (hidden_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    embedding_kernel[(num_m, num_n)](
        input_ids,
        embedding_weight,
        output,
        batch_size,
        seq_len,
        hidden_dim,
        vocab_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return output


@torch.fx.wrap
def fused_embedding_slice_kernel(input_ids, embedding_weight):
    """Fused embedding lookup with slice from position 1."""
    # Use Triton kernel for embedding lookup
    full_embedding = triton_embedding(input_ids, embedding_weight)
    
    # Slice from position 1 (skip the first token)
    sliced_embedding = full_embedding[:, 1:, :]
    
    return sliced_embedding, full_embedding


def pattern(in_0, in_1):
    """Pattern: embedding lookup followed by slice from position 1."""
    embedding = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    sliced = embedding[slice(None, None, None), slice(1, None, None)]
    # Return both values - both must be returned for the pattern to match correctly
    return sliced, embedding


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_embedding_slice_kernel