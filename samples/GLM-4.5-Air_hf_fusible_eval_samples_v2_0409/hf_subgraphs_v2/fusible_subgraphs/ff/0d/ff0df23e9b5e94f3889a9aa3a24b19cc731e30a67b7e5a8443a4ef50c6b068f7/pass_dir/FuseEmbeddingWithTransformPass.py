import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function
def pattern(in_0, in_1):
    """Match embedding lookup with transfer operations"""
    # Match the device transfer and embedding from the original computation
    tmp_1 = in_1.to(device(type='cuda', index=0))
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    """Extract arguments needed for the replacement kernel"""
    return (in_0, in_1)

# Simple Triton kernel for embedding lookup
@triton.jit
def simple_embedding_kernel(
    weights_ptr,      # [num_embeddings, embedding_dim] - embedding weights
    indices_ptr,      # [batch_size] or [H, W] - flattened indices  
    output_ptr,       # [batch_size, embedding_dim] - output embeddings
    num_embeddings,
    embedding_dim,
    output_size,
    weights_stride_0, # stride for first dimension of weights
    weights_stride_1, # stride for second dimension of weights
    output_stride_0,  # stride for first dimension of output
    output_stride_1,  # stride for second dimension of output
    BLOCK_SIZE: tl.constexpr,
):
    """Simple embedding lookup kernel"""
    pid = tl.program_id(0)
    
    # bounds check
    if pid >= output_size:
        return
        
    # Load the index for this position
    idx = tl.load(indices_ptr + pid)
    
    # Load embedding weights for this index
    weights_offset = idx * weights_stride_0
    weights = tl.load(weights_ptr + weights_offset + tl.arange(0, BLOCK_SIZE), 
                      mask=tl.arange(0, BLOCK_SIZE) < embedding_dim, 
                      other=0.0)
    
    # Store the embedding
    output_offset = pid * output_stride_0
    tl.store(output_ptr + output_offset + tl.arange(0, BLOCK_SIZE), 
              weights, 
              mask=tl.arange(0, BLOCK_SIZE) < embedding_dim)
@torch.fx.wrap  
def optimized_embedding_lookup(in_0, in_1):
    """Optimized embedding lookup with device transfer"""
    # Handle device placement for weights
    if in_0.device.type == 'cpu':
        weights = in_0.cuda()
    else:
        weights = in_0
        
    # Move indices to GPU 
    indices = in_1.cuda()
    
    num_embeddings, embedding_dim = weights.shape
    indices_height, indices_width = indices.shape
    
    # Create embedding result with correct shape: [indices_height, indices_width, embedding_dim]
    embedding_result = torch.empty((indices_height, indices_width, embedding_dim), 
                                  device=weights.device, dtype=weights.dtype)
    
    # Get tensor strides
    weights_stride_0 = weights.stride(0)
    weights_stride_1 = weights.stride(1)
    embedding_stride_0 = embedding_result.stride(0)
    embedding_stride_1 = embedding_result.stride(1)
    embedding_stride_2 = embedding_result.stride(2)
    
    # Set block size
    BLOCK_SIZE = 1024
    output_size = indices_height * indices_width
    grid_size = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized kernel to populate embedding tensor
    simple_embedding_kernel[(grid_size,)](
        weights,
        indices.flatten(),
        embedding_result,
        num_embeddings,
        embedding_dim,
        output_size,
        weights_stride_0,
        weights_stride_1,
        embedding_stride_0,
        embedding_stride_2,
        BLOCK_SIZE
    )
    
    return embedding_result



# Replacement function that returns the optimized kernel
def replacement_func():
    return optimized_embedding_lookup