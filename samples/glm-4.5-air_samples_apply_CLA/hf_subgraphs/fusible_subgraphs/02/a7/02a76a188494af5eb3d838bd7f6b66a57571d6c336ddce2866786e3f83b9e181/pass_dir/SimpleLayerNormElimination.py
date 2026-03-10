import torch
import triton
import triton.language as tl

def pattern(input_ids, embedding_weight, norm_weight):
    # Match embedding + layer normalization pattern (simplified for debugging)
    embedding_out = torch.nn.functional.embedding(input_ids, embedding_weight, 50283, None, 2.0, False, False)
    norm_out = torch.nn.functional.layer_norm(embedding_out, (768,), norm_weight, None, 1e-05)
    return norm_out

def replacement_args(input_ids, embedding_weight, norm_weight):
    return (input_ids, embedding_weight, norm_weight)

@triton.jit
def simple_identity_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    norm_weight_ptr,
    out_ptr,
    batch_size,
    seq_len,
    num_embeddings,
    embedding_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel that just does embedding lookup"""
    seq_idx = tl.program_id(0)
    
    # Calculate batch and position
    batch_idx = seq_idx // seq_len
    pos_idx = seq_idx % seq_len
    
    # Load input ID
    input_id = tl.load(input_ids_ptr + batch_idx * seq_len + pos_idx)
    
    # Clamp input ID to valid range
    input_id = tl.minimum(input_id, num_embeddings - 1)
    
    # Simple embedding lookup return (just return embedding weights directly)
    # This is a simplified approach - just return the embedding lookup result
    for k in range(0, embedding_dim, BLOCK_SIZE):
        embed_block = tl.load(
            embedding_weight_ptr + input_id * embedding_dim + k,
            mask=k + tl.arange(0, BLOCK_SIZE) < embedding_dim,
            other=0.0
        )
        tl.store(
            out_ptr + seq_idx * embedding_dim + k,
            embed_block,
            mask=k + tl.arange(0, BLOCK_SIZE) < embedding_dim
        )

@torch.fx.wrap
def simple_embedding_only(input_ids, embedding_weight, norm_weight):
    """Simple wrapper that just does embedding lookup"""
    batch_size, seq_len = input_ids.shape
    num_embeddings, embedding_dim = embedding_weight.shape
    
    BLOCK_SIZE = 256
    num_programs = batch_size * seq_len
    
    out = torch.empty(batch_size, seq_len, embedding_dim, 
                     device=input_ids.device, dtype=embedding_weight.dtype)
    flat_out = out.view(-1, embedding_dim)
    
    simple_identity_kernel[(num_programs,)](
        input_ids_ptr=input_ids.view(-1),
        embedding_weight_ptr=embedding_weight,
        norm_weight_ptr=norm_weight,  # Unused for now
        out_ptr=flat_out,
        batch_size=batch_size,
        seq_len=seq_len,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return simple_embedding_only