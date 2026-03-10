import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    tmp_1 = torch.nn.functional.embedding(in_1, in_0, 1, None, 2.0, False, False)
    tmp_2 = tmp_1 * 1.0
    return (tmp_2,)

def replacement_args(in_1, in_0):
    return (in_1, in_0)

@triton.jit

def simple_embedding_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_size,
    vocab_size,
    embed_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one input element with optimized block processing
    idx = tl.program_id(0)
    
    # Calculate base offset for this input element's output
    output_base = idx * embed_dim
    
    # Read input index
    input_idx = tl.load(input_ptr + idx)
    
    # Calculate base offset for the embedding vector
    embed_base = input_idx * embed_dim
    
    # Process embedding vector efficiently with the given block size
    offsets = output_base + tl.arange(0, BLOCK_SIZE)
    embed_offsets = embed_base + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for both output and weights
    output_mask = offsets < (idx + 1) * embed_dim
    weight_mask = embed_offsets < vocab_size * embed_dim
    
    # Combine masks - load weights only if both are valid
    combined_mask = output_mask & weight_mask
    
    # Load embedding values efficiently
    weight_vals = tl.load(weight_ptr + embed_offsets, mask=combined_mask, other=0.0)
    
    # Store output values
    tl.store(output_ptr + offsets, weight_vals, mask=output_mask)

@torch.fx.wrap
def optimized_embedding_replacement(in_1, in_0):
    # Get tensor shapes
    input_size = in_1.numel()
    vocab_size, embed_dim = in_0.shape
    
    # Determine output shape
    if in_1.dim() == 1:
        output_shape = (input_size, embed_dim)
    else:
        output_shape = in_1.shape + (embed_dim,)
    
    # Ensure both tensors are on the same device
    if in_0.device != in_1.device:
        in_0 = in_0.to(in_1.device)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=in_0.dtype, device=in_1.device)
    
    # Flatten input for easier indexing
    in_1_flat = in_1.flatten()
    
    # Optimized kernel configuration based on input size
    if input_size == 1 and embed_dim <= 1024:
        # For single small input, process entire embedding in one block for efficiency
        BLOCK_SIZE = embed_dim
    else:
        # For larger inputs, use optimal block size
        BLOCK_SIZE = 256
    
    num_programs = input_size
    
    if num_programs > 0 and embed_dim > 0:
        simple_embedding_kernel[(num_programs,)](
            input_ptr=in_1_flat,
            weight_ptr=in_0,
            output_ptr=output.view(-1, embed_dim),
            input_size=input_size,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    # Return the embedding result directly (no multiplication by 1.0)
    return output

def replacement_func():
    return optimized_embedding_replacement