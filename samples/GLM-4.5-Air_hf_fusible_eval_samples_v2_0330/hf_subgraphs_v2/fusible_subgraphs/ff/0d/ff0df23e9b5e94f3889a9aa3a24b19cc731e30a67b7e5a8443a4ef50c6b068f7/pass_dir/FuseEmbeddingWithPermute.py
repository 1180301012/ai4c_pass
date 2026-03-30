import torch
from torch import device
import triton
import triton.language as tl
import math

def pattern(indices, weights):
    # Move indices to GPU
    indices_gpu = indices.to(device(type='cuda', index=0))
    # Embedding lookup
    embedding = torch.nn.functional.embedding(indices_gpu, weights, None, None, 2.0, False, False)
    # Permute [2,0,1] - this converts from [H,W,embedding_dim] to [embedding_dim,H,W]
    permuted = embedding.permute([2, 0, 1])
    return permuted

def replacement_args(indices, weights):
    return (indices, weights)

@triton.jit
def fused_embedding_kernel(
    indices_ptr,
    weights_ptr,
    output_ptr,
    num_embeddings,
    embedding_dim,
    indices_height,
    indices_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one embedding dimension
    embed_dim = tl.program_id(0)
    
    # Calculate start position in weights for this embedding dimension
    weights_offset = embed_dim * num_embeddings
    
    # Output spatial coordinates
    h = tl.program_id(1)
    w = tl.program_id(2)
    
    # Output offset for [embed_dim, h, w]
    output_offset = embed_dim * (indices_height * indices_width) + h * indices_width + w
    
    # Load index for position (h,w)
    index = tl.load(indices_ptr + h * indices_width + w)
    
    # Create mask for valid indices
    mask = index < num_embeddings
    
    # Load weight for this index and embedding dimension with mask
    weight = tl.load(weights_ptr + weights_offset + index, mask=mask, other=0.0)
    
    # Store directly to permuted output [embed_dim, h, w]
    tl.store(output_ptr + output_offset, weight)

@torch.fx.wrap
def fused_embedding_lookup(indices, weights):
    # Get tensor shapes
    indices_shape = indices.shape
    weights_shape = weights.shape
    
    num_embeddings = weights_shape[0]
    embedding_dim = weights_shape[1]
    indices_height = indices_shape[0]
    indices_width = indices_shape[1]
    
    # Output shape: [embedding_dim, indices_height, indices_width]
    output_shape = (embedding_dim, indices_height, indices_width)
    output = torch.empty(output_shape, dtype=weights.dtype, device=weights.device)
    
    # Choose block size - we process one embedding dimension per program
    BLOCK_SIZE = 1  # Each program handles one embedding dimension completely
    
    # Calculate grid dimensions
    num_embed_dims = embedding_dim
    grid = (num_embed_dims, indices_height, indices_width)
    
    # Launch kernel
    fused_embedding_kernel[grid](
        indices_ptr=indices,
        weights_ptr=weights,
        output_ptr=output,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        indices_height=indices_height,
        indices_width=indices_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_embedding_lookup