import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function - must exactly match the computation in the graphs  
def pattern(in_0, in_1):
    # Device transfer, embedding, permute, unsqueeze, expand, contiguous
    # Match the exact structure from model.py including cleanup statements
    tmp_1 = in_1.to(device(type='cuda', index=0));  in_1 = None
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False);  tmp_1 = in_0 = None
    tmp_3 = tmp_2.permute([2, 0, 1]);  tmp_2 = None
    tmp_4 = tmp_3.unsqueeze(0);  tmp_3 = None
    tmp_5 = tmp_4.expand((1, -1, -1, -1));  tmp_4 = None
    tmp_6 = tmp_5.contiguous();  tmp_5 = None
    return (tmp_6,)

# Argument extraction function
def replacement_args(in_0, in_1):
    # Also capture the embedding dimensions for optimization
    embed_dim = in_0.shape[1]
    target_size = in_1.shape[0]
    return (in_0, in_1, embed_dim, target_size)

# Optimized Triton kernel for embedding lookup + expansion (bfloat16 version)
@triton.jit
def embedding_expand_kernel(
    weight_ptr,           # Embedding weights [num_embeddings, embed_dim]
    indices_ptr,          # Indices [seq_len, batch]  
    output_ptr,           # Output [batch_size * target_size, embed_dim]
    num_embeddings,       
    embed_dim,
    seq_len,
    batch_size,
    target_size,
    weight_stride0, weight_stride1,
    indices_stride0, indices_stride1,
    out_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Embedding dimension
    embed_idx = tl.program_id(0)
    # Output position (batch * target_size)
    output_pos = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_pos < batch_size * target_size
    
    if embed_idx >= embed_dim:
        return
        
    # Calculate batch and target position from flattened index
    batch_idx = output_pos // target_size
    target_pos = output_pos % target_size
    
    # Load the actual indices for this batch
    indices = tl.load(indices_ptr + batch_idx * indices_stride1,
                     mask=batch_idx < batch_size, other=0)
    
    # Find which sequence position this target position maps to
    if target_pos < seq_len:
        # Within original sequence length - use actual embedding
        actual_index = indices[target_pos] if target_pos < indices.shape[0] else 0
        actual_index = tl.minimum(actual_index, num_embeddings - 1)
    else:
        # Beyond original sequence length - use zero embedding
        actual_index = 0
    
    # Load embedding weights for this index
    weights = tl.load(weight_ptr + actual_index * weight_stride0 + embed_idx,
                     mask=embed_idx < embed_dim, other=0.0)
    
    # Store the embedding
    tl.store(output_ptr + output_pos * out_stride, weights, mask=mask)

@torch.fx.wrap  
def optimized_embedding_with_postprocessing(in_0, in_1, embed_dim, target_size):
    """Optimized fused embedding with dimension expansion (bfloat16)"""
    # Determine dimensions
    num_embeddings = in_0.shape[0]
    seq_len = in_1.shape[0]
    batch_size = in_1.shape[1]
    
    # Create intermediate flattened output: [batch_size * target_size, embed_dim]
    flattened_shape = (batch_size * target_size, embed_dim)
    flattened_output = torch.empty(flattened_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Calculate strides
    weight_stride0, weight_stride1 = in_0.stride()
    indices_stride0, indices_stride1 = in_1.stride()
    out_stride = flattened_output.stride()[0]
    
    # Kernel launch configuration
    BLOCK_SIZE = 256  # Number of flattened positions processed per program
    
    # Calculate grid dimensions
    grid_m = embed_dim  # One program per embedding dimension
    grid_n = (batch_size * target_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    embedding_expand_kernel[(grid_m, grid_n)](
        in_0, in_1, flattened_output,
        num_embeddings, embed_dim, seq_len, batch_size, target_size,
        weight_stride0, weight_stride1,
        indices_stride0, indices_stride1,
        out_stride,
        BLOCK_SIZE
    )
    
    # Reshape to final output: [1, embed_dim, batch_size, target_size]
    output = flattened_output.reshape(1, embed_dim, batch_size, target_size)
    
    return output

# Replacement function (returns function reference, not called)
def replacement_func():
    return optimized_embedding_with_postprocessing