import torch
import triton
import triton.language as tl

# Pattern matching function - matches a single embedding operation
def pattern(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    return torch.nn.functional.embedding(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

# Argument extraction function
def replacement_args(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    return (input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

# Triton kernel for optimized embedding lookup - simplified version
@triton.jit
def embedding_kernel_simple(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    n_elements,
    vocab_size,
    embed_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data (similar to scalar multiplication)
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Process each element in the block
    for i in range(BLOCK_SIZE):
        if mask[i]:
            input_idx = tl.load(input_ids_ptr + i)
            # Bounds checking for vocabulary
            if 0 <= input_idx < vocab_size:
                # Calculate offset in weight matrix
                weight_offset = input_idx * embed_dim
                # Load the entire embedding vector
                emb_vec = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, embed_dim) < embed_dim, other=0.0)
                # Store in output
                output_offset = i * embed_dim
                tl.store(output_ptr + output_offset + tl.arange(0, embed_dim), emb_vec, mask=tl.arange(0, embed_dim) < embed_dim)

@torch.fx.wrap
def optimized_embedding_forward(input_ids, weight, padding_idx=1, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    # Get dimensions
    input_size = input_ids.numel()
    vocab_size = weight.shape[0]
    embed_dim = weight.shape[1]
    
    # Create output tensor
    output = torch.empty((input_ids.shape[0], embed_dim), dtype=weight.dtype, device=input_ids.device)
    
    # Launch Triton kernel - simplified approach
    BLOCK_SIZE = 256  # Use simpler block size like the working scalar multiplication
    num_programs = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    embedding_kernel_simple[(num_programs,)](
        input_ids_ptr=input_ids,
        weight_ptr=weight,
        output_ptr=output,
        n_elements=input_size,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_embedding_forward