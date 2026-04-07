import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function - test embedding + permute + unsqueeze
def pattern(in_0, in_1):
    """
    Pattern test: embedding + permute + unsqueeze only
    """
    # Embedding lookup (indices first, then weight)
    embedded = torch.nn.functional.embedding(in_1, in_0, None, None, 2.0, False, False)
    
    # Permute dimensions: [embed_dim, batch, seq_len] -> [seq_len, embed_dim, batch]
    permuted = embedded.permute([2, 0, 1])
    
    # Add batch dimension at the beginning
    unsqueezed = permuted.unsqueeze(0)
    
    return unsqueezed

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel for embedding with post-processing
@triton.jit
def optimized_embedding_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    num_embeddings,
    embedding_dim,
    indices_size_0,
    indices_size_1,
    output_size_1,
    output_size_2,
    output_size_3,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one output element
    pid = tl.program_id(0)
    total_elements = output_size_1 * output_size_2 * output_size_3
    
    if pid >= total_elements:
        return
    
    # Calculate output indices: (1, embed_dim, H, W) -> flat index
    offset = pid
    w = offset % output_size_3
    offset //= output_size_3
    h = offset % output_size_2
    offset //= output_size_2
    embed_idx = offset % output_size_1
    batch_idx = offset // output_size_1  # Always 0 for this pattern
    
    # Get index from input tensor (indices are [H, W] or [batch, H, W])
    if indices_size_0 == 1:  # Single batch case
        if indices_size_1 == 1:  # Scalar indices
            idx = tl.load(indices_ptr)
        else:
            idx = tl.load(indices_ptr + h)
    else:  # Multiple batch case - take first batch
        idx = tl.load(indices_ptr + h)
    
    # Clamp index to valid range
    idx = tl.maximum(tl.minimum(idx, num_embeddings - 1), 0)
    
    # Load embedding vector directly to output
    output_offset = (batch_idx * output_size_1 + embed_idx) * output_size_2 * output_size_3 + h * output_size_3 + w
    weight_offset = idx * embedding_dim + embed_idx
    
    # Load one element at a time from embedding vector
    if embed_idx < embedding_dim:
        value = tl.load(weight_ptr + weight_offset)
        tl.store(output_ptr + output_offset, value)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_embedding_lookup(in_0, in_1):
    # Handle device placement - ensure both tensors are on the same device
    if in_1.device != in_0.device:
        in_1 = in_1.to(in_0.device)
    
    # Get input shapes
    num_embeddings = in_0.size(0)
    embedding_dim = in_0.size(1)
    
    # Get indices shape - determines output spatial dimensions
    if len(in_1.shape) == 2:
        indices_H, indices_W = in_1.shape
        indices_total = in_1.size(0) * in_1.size(1)
    else:
        # Handle more complex shapes - flatten spatial dimensions
        indices_total = in_1.numel()
        indices_H = int(indices_total ** 0.5)
        indices_W = indices_total // indices_H
    
    # For bfloat16/float16, the intermediate result after permute+unsqueeze would be:
    # [1, embed_dim, indices_H] where indices_H comes from the spatial dimensions
    
    # Output shape matches what the complete computation would produce: [1, embed_dim, indices_H, indices_W]
    output_shape = (1, embedding_dim, indices_H, indices_W)
    output_size_1, output_size_2, output_size_3 = embedding_dim, indices_H, indices_W
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Calculate grid size
    total_elements = output_size_1 * output_size_2 * output_size_3
    BLOCK_SIZE = 128  # Optimal for this workload
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_embedding_kernel[(num_programs,)](
        indices_ptr=in_1,
        weight_ptr=in_0,
        output_ptr=output,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        indices_size_0=in_1.size(0) if len(in_1.shape) > 2 else 1,
        indices_size_1=in_1.size(-1),
        output_size_1=output_size_1,
        output_size_2=output_size_2,
        output_size_3=output_size_3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
    
    

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_embedding_lookup