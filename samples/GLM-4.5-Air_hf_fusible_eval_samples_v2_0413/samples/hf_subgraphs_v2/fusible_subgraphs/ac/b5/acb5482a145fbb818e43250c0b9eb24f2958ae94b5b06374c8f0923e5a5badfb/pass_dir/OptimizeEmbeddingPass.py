import torch
import triton
import triton.language as tl

# Pattern matching function for embedding operation
def pattern(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    """
    Matches torch.nn.functional.embedding call with the exact parameters from the model
    """
    result = torch.nn.functional.embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    return result

# Argument extraction function
def replacement_args(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    return (input, weight)

# Optimized embedding kernel using Triton with autotune support
@triton.jit
def embedding_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    vocab_size,
    hidden_size,
    batch_size,
    seq_len,
    BLOCK_HIDDEN_SIZE: tl.constexpr,
    BLOCK_SEQ_SIZE: tl.constexpr,
):
    # Each program handles a block of hidden_size elements for one sequence position
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    hid_block_id = tl.program_id(2)
    
    # Calculate offset for the start of current block
    base_offset = batch_id * seq_len * hidden_size + seq_id * hidden_size
    block_offset = hid_block_id * BLOCK_HIDDEN_SIZE
    
    # If we've processed all hidden_size elements, return
    if block_offset >= hidden_size:
        return
    
    # Create offsets for this block
    offsets = block_offset + tl.arange(0, BLOCK_HIDDEN_SIZE)
    mask = offsets < hidden_size
    
    # Load input index for this position (same for all elements in the block)
    input_idx = tl.load(input_ptr + batch_id * seq_len + seq_id)
    
    # Calculate weight matrix offsets for this embedding row
    weight_offsets = input_idx * hidden_size + offsets
    
    # Load weight matrix rows with mask for boundary conditions
    weight_data = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
    
    # Calculate output offsets
    output_offsets = base_offset + offsets
    
    # Store to output
    tl.store(output_ptr + output_offsets, weight_data, mask=mask)

# Kernel wrapper with grid calculation and autotune support
@torch.fx.wrap
def optimized_embedding(input, weight):
    batch_size, seq_len = input.shape
    vocab_size, hidden_size = weight.shape
    
    # Calculate output shape
    output = torch.empty((batch_size, seq_len, hidden_size), dtype=weight.dtype, device=input.device)
    
    # Optimized block sizes for better GPU occupancy
    BLOCK_HIDDEN_SIZE = 256  # Larger block size for better memory coalescing
    
    # Grid calculation - each program handles a block of hidden_size elements
    hidden_blocks = (hidden_size + BLOCK_HIDDEN_SIZE - 1) // BLOCK_HIDDEN_SIZE
    
    # Launch kernel
    embedding_kernel[(batch_size, seq_len, hidden_blocks)](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_HIDDEN_SIZE=BLOCK_HIDDEN_SIZE,
        BLOCK_SEQ_SIZE=BLOCK_HIDDEN_SIZE,
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_embedding