import torch
import triton
import triton.language as tl

# Pattern matching function - optimize expand operation specifically  
def pattern(weight, hidden_states, key_states):
    tmp_4 = key_states[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_5 = tmp_4.expand(hidden_states.shape[0], 4, 4, 
                        key_states.shape[-2] if len(key_states.shape) == 4 else (hidden_states.shape[1] // (hidden_states.shape[2] // 128)), 
                        128)
    return tmp_5

# Argument extraction function  
def replacement_args(weight, hidden_states, key_states):
    return (weight, hidden_states, key_states)

# Optimized slice-expand operation using Triton
@triton.jit
def slice_expand_kernel(
    key_states_ptr,
    out_ptr,
    key_dim1, key_dim2, reshape_dim, head_dim,
    batch_size, expand_dim2,
    BLOCK_BATCH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Compute ranges for this program
    batch_offset = pid * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    
    # Map to key_states indices - output has shape [batch_size, expand_dim2, expand_dim2, reshape_dim, head_dim]
    # We're mapping from key_states [key_dim1, key_dim2, reshape_dim, head_dim]
    # For all the patterns, we need to broadcast appropriately
    
    # Determine how to map batch and expand dimensions to key_states dimensions
    if key_dim1 == batch_size and key_dim2 == expand_dim2:
        # Pattern: key_states[batch, expand_dim, ...] -> expand along expand_dim
        key1_indices = batch_offset // expand_dim2  # batch dimension
        key2_indices = batch_offset % expand_dim2    # first expand dimension
        key1_mask = key1_indices < batch_size
        key2_mask = key2_indices < expand_dim2
    elif key_dim1 == 1 and key_dim2 == expand_dim2:
        # Pattern: key_states[1, expand_dim, ...] -> broadcast batch dimension
        key1_indices = torch.zeros_like(batch_offset)  # always 0, broadcast to batch_size
        key2_indices = batch_offset % expand_dim2
        key1_mask = key1_indices == 0  # always True for broadcasting
        key2_mask = batch_offset < expand_dim2
    else:
        # Fallback for other patterns
        key1_indices = torch.zeros_like(batch_offset)
        key2_indices = batch_offset % expand_dim2
        key1_mask = key1_indices == 0
        key2_mask = batch_offset < expand_dim2
    
    # Key data indices (reshape_dim, head_dim)
    reshape_indices = tl.arange(0, reshape_dim)
    head_indices = tl.arange(0, HEAD_DIM)
    
    # Create masks
    reshape_mask = reshape_indices < reshape_dim
    head_mask = head_indices < HEAD_DIM
    
    # Load from key_states [dim1, dim2, reshape_dim, head_dim]
    key_ptrs = key_states_ptr + (
        key1_indices[:, None, None, None] * (key_dim2 * reshape_dim * HEAD_DIM) +
        key2_indices[None, :, None, None] * (reshape_dim * HEAD_DIM) +
        reshape_indices[None, None, :, None] * HEAD_DIM +
        head_indices[None, None, None, :]
    )
    
    # Load key_states data  
    key_data = tl.load(key_ptrs, 
                      mask=key1_mask[:, None, None, None] & 
                           key2_mask[None, :, None, None] & 
                           reshape_mask[None, None, :, None] & 
                           head_mask[None, None, None, :], 
                      other=0.0)
    
    # Store in expanded format [batch_size, expand_dim2, expand_dim2, reshape_dim, head_dim]
    # We need to broadcast along the second expand_dim
    expand_indices = tl.arange(0, expand_dim2)
    out_ptrs = out_ptr + (
        batch_offset[:, None, None, None, None] * (expand_dim2 * expand_dim2 * reshape_dim * HEAD_DIM) +
        expand_indices[None, None, None, None, :] * (expand_dim2 * reshape_dim * HEAD_DIM) +
        key2_indices[None, None, None, :, None] * (reshape_dim * HEAD_DIM) +
        reshape_indices[None, None, :, None, None] * HEAD_DIM +
        head_indices[None, None, None, None, :]
    )
    
    # Broadcast to all expand dimensions
    for i in range(expand_dim2):
        current_out_ptrs = out_ptrs + i * (reshape_dim * HEAD_DIM)
        current_mask = batch_offset < batch_size
        tl.store(current_out_ptrs, key_data, mask=current_mask[:, None, None, None, None] & 
                                            head_mask[None, None, None, None, :])

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_slice_expand(weight, hidden_states, key_states):
    # Get tensor shapes
    if len(key_states.shape) == 4:
        key_dim1, key_dim2, reshape_dim, head_dim = key_states.shape
    else:
        raise ValueError(f"Unsupported key_states shape: {key_states.shape}")
    
    batch_size = hidden_states.shape[0]
    expand_dim2 = 4  # This is fixed at 4 for all patterns
    
    # Create output tensor [batch_size, expand_dim2, expand_dim2, reshape_dim, head_dim]
    output = torch.empty((batch_size, expand_dim2, expand_dim2, reshape_dim, head_dim), 
                        dtype=key_states.dtype, 
                        device=key_states.device)
    
    # Calculate grid dimensions
    BLOCK_BATCH = 64
    grid_batch = (batch_size * expand_dim2 + BLOCK_BATCH - 1) // BLOCK_BATCH
    
    # Launch kernel
    slice_expand_kernel[grid_batch](
        key_states,
        output,
        key_dim1, key_dim2, reshape_dim, head_dim,
        batch_size, expand_dim2,
        BLOCK_BATCH,
        head_dim,
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_slice_expand