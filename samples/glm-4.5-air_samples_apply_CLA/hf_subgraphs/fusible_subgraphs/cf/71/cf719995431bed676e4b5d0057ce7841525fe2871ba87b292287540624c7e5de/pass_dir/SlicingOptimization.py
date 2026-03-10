import torch
import triton
import triton.language as tl

# Pattern: tensor slicing operation
def pattern(input_tensor):
    tmp_2 = input_tensor[slice(None, None, None), slice(None, None, None), 1]
    return tmp_2

# Extract arguments for the replacement function
def replacement_args(input_tensor):
    return (input_tensor,)

# Triton kernel for optimized slicing
@triton.jit
def slicing_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    n_seq_full,
    n_embed,
    n_seq_sliced,
    dim_to_slice,
    slice_idx,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_EMBED: tl.constexpr,
):
    # Each program handles one batch x sequence position combination
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    embed_offset = tl.arange(0, BLOCK_SIZE_EMBED)
    
    # Calculate input and output offsets
    full_offset = batch_idx * n_seq_full * n_embed + seq_idx * n_embed + embed_offset
    sliced_offset = batch_idx * n_seq_sliced * n_embed + seq_idx * n_embed + embed_offset
    
    # Calculate the actual sequence index in the full tensor
    actual_seq_idx = seq_idx if dim_to_slice != 1 else slice_idx
    
    # Validate the sequence index
    if dim_to_slice == 1 and seq_idx < n_seq_sliced:
        # For dim=1 slicing, we need to check if the source index is valid
        source_offset = batch_idx * n_seq_full * n_embed + actual_seq_idx * n_embed + embed_offset
        
        # Load input data with bounds checking
        input_mask = (source_offset < (n_batch * n_seq_full * n_embed)) & (embed_offset < n_embed)
        input_data = tl.load(input_ptr + source_offset, mask=input_mask, other=0.0)
        
        # Store output data with bounds checking
        output_mask = (sliced_offset < (n_batch * n_seq_sliced * n_embed)) & (embed_offset < n_embed)
        tl.store(output_ptr + sliced_offset, input_data, mask=output_mask)

@torch.fx.wrap
def optimized_slicing(input_tensor, slice_dim=1, slice_idx=1):
    # Get input shape
    input_shape = input_tensor.shape
    if len(input_shape) == 3:  # Expected shape: [batch, seq, embed]
        n_batch, n_seq_full, n_embed = input_shape
    else:
        # Fallback for different shapes
        return input_tensor[slice(None), slice(None), slice_idx]
    
    if slice_dim == 1:
        n_seq_sliced = n_seq_full - 1  # Slicing from index 1 removes first element
    else:
        n_seq_sliced = n_seq_full
    
    # Create output tensor
    if slice_dim == 1:
        output_shape = (n_batch, n_seq_sliced, n_embed)
    else:
        output_shape = (n_batch, n_seq_full, n_embed)
    
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set up grid dimensions
    batch_blocks = (n_batch + 63) // 64
    seq_blocks = (max(n_seq_full, n_seq_sliced) + 63) // 64
    
    # Launch kernel
    slicing_kernel[(batch_blocks, seq_blocks, 1)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_batch=n_batch,
        n_seq_full=n_seq_full,
        n_embed=n_embed,
        n_seq_sliced=n_seq_sliced,
        dim_to_slice=slice_dim,
        slice_idx=slice_idx,
        BLOCK_SIZE_BATCH=64,
        BLOCK_SIZE_SEQ=64,
        BLOCK_SIZE_EMBED=128,
    )
    
    return output

def replacement_func():
    return optimized_slicing