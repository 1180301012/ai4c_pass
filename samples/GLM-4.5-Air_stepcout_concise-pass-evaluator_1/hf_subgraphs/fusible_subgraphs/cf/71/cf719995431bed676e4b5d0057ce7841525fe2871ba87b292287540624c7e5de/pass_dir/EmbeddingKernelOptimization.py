import torch
import triton
import triton.language as tl

def pattern(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, **kwargs):
    """
    Pattern to match: embedding operation (no slice)
    """
    # Embedding operation
    embedded = torch.nn.functional.embedding(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    return embedded

def replacement_args(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, **kwargs):
    """
    Extract arguments needed for the optimized embedding operation
    """
    input_ids = input_ids.contiguous()
    weight = weight.contiguous()
    
    return (input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

@triton.jit
def optimized_embedding_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    input_ids_size,
    embedding_dim,
    vocab_size,
    padding_idx,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Optimized embedding kernel with better memory access patterns
    """
    # Get program IDs
    pid = tl.program_id(0)
    
    # Calculate which block we're processing
    block_start = pid * BLOCK_SIZE
    end_idx = min(block_start + BLOCK_SIZE, input_ids_size)
    
    # Process this block
    for i in range(block_start, end_idx):
        # Load input ID
        input_id = tl.load(input_ids_ptr + i, other=padding_idx)
        
        # Calculate pointer to embedding vector
        emb_base = weight_ptr + input_id * embedding_dim
        
        # Load embedding vector efficiently
        for j in range(0, embedding_dim, BLOCK_SIZE_M):
            # Current block of this embedding
            m = min(BLOCK_SIZE_M, embedding_dim - j)
            
            # Create offset mask
            offsets = j + tl.arange(0, m)
            
            # Load from weight matrix
            emb_vals = tl.load(emb_base + offsets, other=0.0)
            
            # Store to output
            tl.store(output_ptr + i * embedding_dim + offsets, emb_vals)

@triton.jit
def optimized_embedding_kernel_vec(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    input_ids_size,
    embedding_dim,
    vocab_size,
    padding_idx,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized embedding kernel using vectorized loads for better performance
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_ids_size
    
    # Load input IDs
    input_ids = tl.load(input_ids_ptr + offsets, mask=mask, other=padding_idx)
    
    # Process each input ID in this block
    emb_ptr_base = tl.make_block_ptr(
        base_ptr=weight_ptr,
        shape=(vocab_size, embedding_dim),
        strides=(embedding_dim, 1),
        block_shape=(1, embedding_dim),
        order=(1, 0)
    )
    
    # For each input ID, load its embedding
    for emb_idx in range(BLOCK_SIZE):
        if offsets[emb_idx] < input_ids_size:
            input_id = input_ids[emb_idx]
            if input_id >= 0:  # Not padding
                # Pointer to this embedding
                emb_ptr = emb_ptr_base + input_id
                # Load the entire embedding vector
                emb_vec = tl.load(emb_ptr, mask=tl.arange(0, embedding_dim) < embedding_dim, other=0.0)
                # Store to output
                out_base = output_ptr + offsets[emb_idx] * embedding_dim
                tl.store(out_base + tl.arange(0, embedding_dim), emb_vec, mask=tl.arange(0, embedding_dim) < embedding_dim)

@torch.fx.wrap
def optimized_embedding_function(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    """
    Optimized embedding function using Triton
    """
    input_ids_size = input_ids.numel()
    embedding_dim = weight.size(1)
    vocab_size = weight.size(0)
    
    # Create output tensor
    output = torch.empty(input_ids_size, embedding_dim, dtype=weight.dtype, device=weight.device)
    
    # Choose kernel and block sizes based on input size
    if input_ids_size < 1024:
        # Small input - use vectorized kernel
        BLOCK_SIZE = 128
        num_programs = (input_ids_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        optimized_embedding_kernel_vec[(num_programs,)](
            input_ids_ptr=input_ids,
            weight_ptr=weight,
            output_ptr=output,
            input_ids_size=input_ids_size,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Larger input - use block processing kernel
        BLOCK_SIZE = 1024
        BLOCK_SIZE_M = 128  # Vector size for memory loads
        num_programs = (input_ids_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        optimized_embedding_kernel[(num_programs,)](
            input_ids_ptr=input_ids,
            weight_ptr=weight,
            output_ptr=output,
            input_ids_size=input_ids_size,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
        )
    
    # Handle normalization if needed (simplified - in real implementation would need proper normalization)
    if max_norm is not None:
        # Clip L2 norm
        output_norm = torch.norm(output, p=2, dim=1, keepdim=True)
        output = torch.where(output_norm > max_norm, output * max_norm / output_norm, output)
    
    return output

def replacement_func():
    """
    Return the optimized embedding function
    """
    return optimized_embedding_function