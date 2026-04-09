import torch
import triton
import triton.language as tl

def pattern(input_ids, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """Match torch.nn.functional.embedding operation with exact signature from model.py"""
    return torch.nn.functional.embedding(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

def replacement_args(input_ids, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """Extract arguments for replacement function"""
    return (input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

@triton.jit
def embedding_lookup_kernel(
    indices_ptr,      # Pointer to input IDs tensor
    weight_ptr,       # Pointer to embedding table
    output_ptr,       # Pointer to output tensor
    num_embeddings,   # Number of embeddings (151936)
    embedding_dim,    # Embedding dimension (1536)
    batch_size,       # Batch size
    seq_len,          # Sequence length
    BLOCK_SIZE: tl.constexpr,
):
    """Triton-optimized embedding lookup kernel"""
    # Each program handles one element in the input tensor
    pid = tl.program_id(0)
    total_elements = batch_size * seq_len
    
    if pid >= total_elements:
        return
    
    # Compute batch and sequence indices
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Load input index
    input_index = tl.load(indices_ptr + pid)
    
    # Check if index is valid
    if input_index < num_embeddings:
        # Compute embedding start position: input_index * embedding_dim
        emb_start = input_index * embedding_dim
        
        # Process embedding dimensions in chunks
        dim_idx = 0
        while dim_idx < embedding_dim:
            # Load one element from embedding vector
            emb_pos = emb_start + dim_idx
            embed_val = tl.load(weight_ptr + emb_pos)
            
            # Store in output
            out_pos = pid * embedding_dim + dim_idx
            tl.store(output_ptr + out_pos, embed_val)
            
            dim_idx += 1

@torch.fx.wrap
def triton_embedding(input_ids, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """High-performance embedding lookup using Triton"""
    # Get tensor shapes
    batch_size, seq_len = input_ids.shape
    num_embeddings, embedding_dim = weight.shape
    
    # Check input types (convert if needed)
    if input_ids.dtype != torch.int64:
        input_ids = input_ids.long()
    
    # Allocate output tensor with correct dtype
    output = torch.empty((batch_size, seq_len, embedding_dim), 
                        dtype=weight.dtype,
                        device=input_ids.device)
    
    total_elements = batch_size * seq_len
    BLOCK_SIZE = 1024  # Fixed block size for simplicity
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with autotuning
    embedding_lookup_kernel[(num_programs,)](
        input_ids, weight, output,
        num_embeddings, embedding_dim, batch_size, seq_len, BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """Return the optimized embedding function"""
    return triton_embedding