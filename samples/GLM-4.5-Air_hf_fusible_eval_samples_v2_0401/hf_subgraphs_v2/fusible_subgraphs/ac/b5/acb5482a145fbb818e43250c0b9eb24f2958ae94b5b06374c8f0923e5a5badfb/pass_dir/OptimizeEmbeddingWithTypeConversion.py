import torch
import triton
import triton.language as tl

def pattern(input_ids, embedding_weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    """Simple pattern matching embedding lookup only"""
    return torch.nn.functional.embedding(input_ids, embedding_weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

def replacement_args(input_ids, embedding_weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    return (input_ids, embedding_weight)

@triton.jit
def simple_embedding_kernel(
    input_ids_ptr,
    embedding_ptr,
    output_ptr,
    num_embeddings,
    embedding_dim,
    batch_size,
    seq_len,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Simple 3D embedding lookup kernel"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Triton doesn't support chained boolean operators, so check separately
    if pid_m >= batch_size:
        return
    if pid_n >= seq_len:
        return
    if pid_k >= embedding_dim:
        return
    
    # Load input ID
    input_id = tl.load(input_ids_ptr + pid_m * seq_len + pid_n)
    
    # Handle invalid indices
    if input_id < 0 or input_id >= num_embeddings:
        tl.store(output_ptr + pid_m * seq_len * embedding_dim + pid_n * embedding_dim + pid_k, 0.0)
        return
    
    # Load embedding value
    embed_idx = input_id * embedding_dim + pid_k
    embed_val = tl.load(embedding_ptr + embed_idx, mask=embed_idx < (num_embeddings * embedding_dim), other=0.0)
    tl.store(output_ptr + pid_m * seq_len * embedding_dim + pid_n * embedding_dim + pid_k, embed_val)



@torch.fx.wrap
def triton_embedding(input_ids, embedding_weight):
    """Triton optimized embedding lookup"""
    batch_size, seq_len = input_ids.shape
    num_embeddings, embedding_dim = embedding_weight.shape
    
    output = torch.empty((batch_size, seq_len, embedding_dim), dtype=embedding_weight.dtype, device=input_ids.device)
    
    # Optimized block sizes for better performance
    BLOCK_M = 8   # Batch items per program
    BLOCK_N = 32  # Sequence items per program  
    BLOCK_K = 64  # Embedding dims per program
    
    grid_m = (batch_size + BLOCK_M - 1) // BLOCK_M
    grid_n = (seq_len + BLOCK_N - 1) // BLOCK_N
    grid_k = (embedding_dim + BLOCK_K - 1) // BLOCK_K
    
    simple_embedding_kernel[(grid_m, grid_n, grid_k)](
        input_ids, embedding_weight, output,
        num_embeddings, embedding_dim, batch_size, seq_len,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return output

def replacement_func():
    return triton_embedding