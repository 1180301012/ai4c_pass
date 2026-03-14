import torch
import triton
import triton.language as tl

def pattern(tmp_5, weight, in_3):
    # The pattern matches from index creation to addition result
    # tmp_5 should be [1, 3] with values [3, 3]
    embedded = torch.nn.functional.embedding(tmp_5, weight, None, None, 2.0, False, False)
    result = in_3 + embedded
    return result

def replacement_args(tmp_5, weight, in_3):
    return (tmp_5, weight, in_3)

@triton.jit
def fused_embedding_add_kernel(
    indices_ptr,
    weight_ptr,
    input_embeds_ptr,
    out_ptr,
    num_embeddings,
    embedding_dim,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Embedding is expensive, we index into the weight matrix
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return
    
    # Load index (should be constant 3 based on our analysis)
    index = tl.load(indices_ptr + batch_idx * seq_len + seq_idx)
    
    # Calculate offset in weight matrix
    weight_offset = index * embedding_dim
    
    # Load embedding from weight matrix
    embed_offset = tl.arange(0, BLOCK_SIZE)
    mask = embed_offset < embedding_dim
    embedding = tl.load(weight_ptr + weight_offset + embed_offset, mask=mask, other=0.0)
    
    # Load input embedding
    input_offset = (batch_idx * seq_len + seq_idx) * embedding_dim + embed_offset
    input_embed = tl.load(input_embeds_ptr + input_offset, mask=mask, other=0.0)
    
    # Add input embedding to the learned embedding
    result = embedding + input_embed
    
    # Store result
    out_offset = (batch_idx * seq_len + seq_idx) * embedding_dim + embed_offset
    tl.store(out_ptr + out_offset, result, mask=mask)

@torch.fx.wrap
def fused_embedding_add(tmp_5, weight, in_3):
    batch_size, seq_len, embedding_dim = in_3.shape
    
    # Handle single batch, sequence length 3, embedding dim 1024
    BLOCK_SIZE = 1024
    num_programs = batch_size * seq_len
    
    # Create output tensor
    out = torch.empty_like(in_3)
    
    # Use tmp_5 directly as indices (should be [[3, 3]])
    flat_indices = tmp_5.flatten()
    
    fused_embedding_add_kernel[(num_programs,)](
        indices_ptr=flat_indices,
        weight_ptr=weight,
        input_embeds_ptr=in_3,
        out_ptr=out,
        num_embeddings=weight.size(0),
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_embedding_add