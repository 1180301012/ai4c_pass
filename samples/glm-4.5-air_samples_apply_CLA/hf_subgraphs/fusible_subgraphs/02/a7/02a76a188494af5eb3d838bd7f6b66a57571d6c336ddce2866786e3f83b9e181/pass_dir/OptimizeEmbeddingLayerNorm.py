import torch
import triton
import triton.language as tl

def pattern(input_ids, embedding_weight, norm_weight):
    # Match embedding + layer normalization pattern
    embedding_out = torch.nn.functional.embedding(input_ids, embedding_weight, 50283, None, 2.0, False, False)
    norm_out = torch.nn.functional.layer_norm(embedding_out, (768,), norm_weight, None, 1e-05)
    return norm_out

def replacement_args(input_ids, embedding_weight, norm_weight):
    return (input_ids, embedding_weight, norm_weight)

@triton.jit
def embedding_lookup_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    embedding_out_ptr,
    num_embeddings,
    embedding_dim,
    batch_size,
    seq_len,
    BLOCK_SIZE_EMBEDDING: tl.constexpr,
):
    """Optimized embedding lookup kernel"""
    seq_idx = tl.program_id(0)
    
    # Calculate batch and position
    batch_idx = seq_idx // seq_len
    pos_idx = seq_idx % seq_len
    
    # Load input ID
    input_id = tl.load(input_ids_ptr + batch_idx * seq_len + pos_idx)
    
    # Clamp input ID to valid range
    input_id = tl.minimum(input_id, num_embeddings - 1)
    
    # Initialize output
    out_sum = 0.0
    out_sum_sq = 0.0
    
    # Process embedding dimension with loop unrolling
    for k in range(0, embedding_dim, BLOCK_SIZE_EMBEDDING):
        # Load embedding block
        embed_block = tl.load(
            embedding_weight_ptr + input_id * embedding_dim + k,
            mask=k + tl.arange(0, BLOCK_SIZE_EMBEDDING) < embedding_dim,
            other=0.0
        )
        
        # Store embedding block
        tl.store(
            embedding_out_ptr + seq_idx * embedding_dim + k,
            embed_block,
            mask=k + tl.arange(0, BLOCK_SIZE_EMBEDDING) < embedding_dim
        )
        
        # Accumulate statistics for layer norm
        out_sum += tl.sum(embed_block)
        out_sum_sq += tl.sum(embed_block * embed_block)

@triton.jit
def layer_norm_kernel(
    embedding_out_ptr,
    norm_weight_ptr,
    layer_norm_out_ptr,
    embedding_dim,
    seq_len,
    batch_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized layer normalization kernel"""
    seq_idx = tl.program_id(0)
    
    # Compute mean and variance
    out_sum = 0.0
    out_sum_sq = 0.0
    
    # Read all embedding values for this token to compute mean/var
    for k in range(0, embedding_dim, BLOCK_SIZE):
        embed_block = tl.load(
            embedding_out_ptr + seq_idx * embedding_dim + k,
            mask=k + tl.arange(0, BLOCK_SIZE) < embedding_dim,
            other=0.0
        )
        out_sum += tl.sum(embed_block)
        out_sum_sq += tl.sum(embed_block * embed_block)
    
    # Normalize
    N = float(embedding_dim)
    mean = out_sum / N
    var = out_sum_sq / N - mean * mean
    
    # Apply normalization with weights
    for k in range(0, embedding_dim, BLOCK_SIZE):
        # Load embedding and weight
        embed_val = tl.load(
            embedding_out_ptr + seq_idx * embedding_dim + k,
            mask=k + tl.arange(0, BLOCK_SIZE) < embedding_dim,
            other=0.0
        )
        weight_val = tl.load(
            norm_weight_ptr + k,
            mask=k + tl.arange(0, BLOCK_SIZE) < embedding_dim,
            other=1.0
        )
        
        # Apply layer normalization formula
        normalized_val = (embed_val - mean) / tl.sqrt(var + eps) * weight_val
        
        # Store result
        tl.store(
            layer_norm_out_ptr + seq_idx * embedding_dim + k,
            normalized_val,
            mask=k + tl.arange(0, BLOCK_SIZE) < embedding_dim
        )

@torch.fx.wrap
def optimized_embedding_layer_norm(input_ids, embedding_weight, norm_weight):
    """Optimized wrapper combining embedding and layer normalization"""
    batch_size, seq_len = input_ids.shape
    num_embeddings, embedding_dim = embedding_weight.shape
    
    # Handle different input sizes across the three graphs
    if seq_len == 64:
        # Graph 0: [1, 64]
        BLOCK_SIZE_EMBEDDING = 256
        BLOCK_SIZE_LAYER_NORM = 256
    elif seq_len == 512:
        # Graph 5: [4, 512]
        BLOCK_SIZE_EMBEDDING = 128
        BLOCK_SIZE_LAYER_NORM = 128
    else:
        # Graph 7: [64, 128]
        BLOCK_SIZE_EMBEDDING = 64
        BLOCK_SIZE_LAYER_NORM = 64
    
    # Flatten input for kernel launch
    flat_input_ids = input_ids.view(-1)  # [batch_size * seq_len]
    
    # Create intermediate and output tensors
    embedding_out = torch.empty(batch_size * seq_len, embedding_dim, 
                               device=input_ids.device, dtype=embedding_weight.dtype)
    layer_norm_out = torch.empty_like(embedding_out)
    
    # Launch embedding lookup kernel
    num_programs = batch_size * seq_len
    embedding_lookup_kernel[(num_programs,)](
        input_ids_ptr=flat_input_ids,
        embedding_weight_ptr=embedding_weight,
        embedding_out_ptr=embedding_out,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE_EMBEDDING=BLOCK_SIZE_EMBEDDING,
    )
    
    # Launch layer normalization kernel
    layer_norm_kernel[(num_programs,)](
        embedding_out_ptr=embedding_out,
        norm_weight_ptr=norm_weight,
        layer_norm_out_ptr=layer_norm_out,
        embedding_dim=embedding_dim,
        seq_len=seq_len,
        batch_size=batch_size,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE_LAYER_NORM,
    )
    
    # Reshape output to match expected shape
    return layer_norm_out.view(batch_size, seq_len, embedding_dim)

def replacement_func():
    return optimized_embedding_layer_norm