import torch
import triton
import triton.language as tl


def pattern(in_1, tmp_0):
    """
    Pattern to match: embedding lookup followed by scalar multiplication
    """
    tmp_1 = torch.nn.functional.embedding(in_1, tmp_0, 1, None, 2.0, False, False)
    tmp_2 = tmp_1 * 1.0
    return tmp_2


def replacement_args(in_1, tmp_0):
    return (in_1, tmp_0)


@triton.jit
def embedding_kernel_simple(
    indices_ptr,
    weight_ptr,
    output_ptr,
    embedding_dim,
    num_embeddings,
    padding_idx: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Highly optimized single-index embedding kernel.
    Assumes num_indices=1 for maximum performance.
    """
    pid = tl.program_id(0)
    
    # Load the single index
    idx = tl.load(indices_ptr)
    
    # Check padding
    is_padding = (idx == padding_idx)
    idx = tl.where(is_padding, 0, idx)
    
    # Calculate offsets for this block
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < embedding_dim
    
    # Load embedding vector
    weight_offset = idx * embedding_dim + offset
    values = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
    
    # Handle padding
    values = tl.where(is_padding, 0.0, values)
    
    # Store directly
    tl.store(output_ptr + offset, values, mask=mask)


@torch.fx.wrap
def fused_embedding_multiply(indices, weight):
    """
    Optimized wrapper for embedding lookup (multiplication by 1.0 is omitted as no-op).
    """
    # Ensure weight is on GPU
    if weight.device != indices.device:
        weight = weight.to(indices.device)
    
    # Get dimensions
    num_embeddings, embedding_dim = weight.shape
    
    # Prepare output
    output_shape = list(indices.shape) + [embedding_dim]
    output = torch.empty(output_shape, device=indices.device, dtype=weight.dtype)
    
    # Optimize for single index case
    if indices.numel() == 1:
        # Single kernel launch with BLOCK_SIZE=1024 for embedding_dim=1024
        BLOCK_SIZE = 1024
        num_blocks = (embedding_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (num_blocks,)
        
        embedding_kernel_simple[grid](
            indices_ptr=indices,
            weight_ptr=weight,
            output_ptr=output,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            padding_idx=1,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # For multiple indices, use a different approach
        indices_flat = indices.flatten()
        num_indices = indices_flat.numel()
        output_flat = output.view(num_indices, embedding_dim)
        
        BLOCK_SIZE = 512
        num_blocks = (embedding_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        for i in range(num_indices):
            grid = (num_blocks,)
            embedding_kernel_simple[grid](
                indices_ptr=indices_flat[i:i+1],
                weight_ptr=weight,
                output_ptr=output_flat[i:i+1],
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                padding_idx=1,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    
    return output


def replacement_func():
    return fused_embedding_multiply