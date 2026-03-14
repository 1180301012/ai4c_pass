import torch
import triton
import triton.language as tl

def pattern(key_states, linear_output):
    # Slice operation that adds a new dimension at position 2
    tmp_4 = key_states[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    
    # Expand operation - determine target shape from key_states shape + model knowledge
    original_shape = key_states.shape  # [batch_size, num_heads, seq_len_per_head, head_dim]
    batch_size, num_heads, seq_len_per_head, head_dim = original_shape
    
    # Target expand shape: [batch_size, num_heads, num_expands, seq_len_per_head, head_dim]
    # All graphs expand the 3rd dimension from 1 to 4
    num_expands = 4
    expand_shape = (batch_size, num_heads, num_expands, seq_len_per_head, head_dim)
    
    tmp_5 = tmp_4.expand(expand_shape)
    
    return tmp_5

def replacement_args(key_states, linear_output):
    return (key_states, linear_output)

@triton.jit
def slice_expand_kernel(
    key_states_ptr,
    output_ptr,
    batch_size,
    num_heads,
    seq_len_per_head,
    head_dim,
    num_expands,
    
    BATCH_TILE: tl.constexpr,
    HEADS_TILE: tl.constexpr,
    SEQ_TILE: tl.constexpr,
    HEAD_DIM_TILE: tl.constexpr,
    EXPANDS_TILE: tl.constexpr,
):
    # Calculate program IDs
    pid_batch = tl.cdiv(batch_size, BATCH_TILE)
    pid_heads = tl.cdiv(num_heads, HEADS_TILE)
    pid_seq = tl.cdiv(seq_len_per_head, SEQ_TILE)
    pid_head_dim = tl.cdiv(head_dim, HEAD_DIM_TILE)
    pid_expands = tl.cdiv(num_expands, EXPANDS_TILE)
    
    # Parallelize over all dimensions
    for bid in range(tl.program_id(0)):
        if bid >= pid_batch:
            break
        for hid in range(tl.program_id(1)):
            if hid >= pid_heads:
                break
        for sid in range(tl.program_id(2)):
            if sid >= pid_seq:
                break
        for hdim in range(tl.program_id(3)):
            if hdim >= pid_head_dim:
                break
        for exp in range(tl.program_id(4)):
            if exp >= pid_expands:
                break
            
            # Calculate indices for input (4D) and output (5D)
            batch_idx = bid * BATCH_TILE + tl.arange(0, BATCH_TILE)
            head_idx = hid * HEADS_TILE + tl.arange(0, HEADS_TILE)
            seq_idx = sid * SEQ_TILE + tl.arange(0, SEQ_TILE)
            head_dim_idx = hdim * HEAD_DIM_TILE + tl.arange(0, HEAD_DIM_TILE)
            expand_idx = exp * EXPANDS_TILE + tl.arange(0, EXPANDS_TILE)
            
            # Create masks
            batch_mask = batch_idx < batch_size
            head_mask = head_idx < num_heads
            seq_mask = seq_idx < seq_len_per_head
            head_dim_mask = head_dim_idx < head_dim
            expand_mask = expand_idx < num_expands
            
            # Load from 4D key_states: [batch_size, num_heads, seq_len_per_head, head_dim]
            input_start = batch_idx[:, None, None, None] * num_heads * seq_len_per_head * head_dim + \
                         head_idx[None, :, None, None] * seq_len_per_head * head_dim + \
                         seq_idx[None, None, :, None] * head_dim + \
                         head_dim_idx[None, None, None, :]
            
            input_mask = batch_mask[:, None, None, None] & head_mask[None, :, None, None] & seq_mask[None, None, :, None] & head_dim_mask[None, None, None, :]
            
            input_data = tl.load(key_states_ptr + input_start, mask=input_mask, other=0.0)
            
            # Broadcast to 5D output: [batch_size, num_heads, num_expands, seq_len_per_head, head_dim]
            # For each expand dimension, we copy the same data
            output_start = batch_idx[:, None, None, None, None] * num_heads * num_expands * seq_len_per_head * head_dim + \
                          head_idx[None, :, None, None, None] * num_expands * seq_len_per_head * head_dim + \
                          expand_idx[None, None, :, None, None] * seq_len_per_head * head_dim + \
                          seq_idx[None, None, None, :, None] * head_dim + \
                          head_dim_idx[None, None, None, None, :]
            
            output_mask = batch_mask[:, None, None, None, None] & head_mask[None, :, None, None, None] & expand_mask[None, None, :, None, None] & seq_mask[None, None, None, :, None] & head_dim_mask[None, None, None, None, :]
            
            # Broadcast: the same input_data is written for all expand indices
            input_expanded = input_data[:, :, None, :, :]  # Add expand dimension
            input_reshaped = input_expanded.reshape(-1)  # Flatten for broadcasting
            
            # Reshape output indices for broadcasting
            flat_output = output_start.reshape(-1)
            flat_output_mask = output_mask.reshape(-1)
            
            # Broadcast the data
            tl.store(output_ptr + flat_output, input_reshaped, mask=flat_output_mask)

@torch.fx.wrap
def slice_expand_fused(key_states, linear_output):
    # Get key_states shape
    original_shape = key_states.shape
    batch_size, num_heads, seq_len_per_head, head_dim = original_shape
    num_expands = 4  # All graphs expand to 4 in the 3rd dimension
    
    # Output shape: [batch_size, num_heads, num_expands, seq_len_per_head, head_dim]
    expand_shape = (batch_size, num_heads, num_expands, seq_len_per_head, head_dim)
    output = torch.empty(expand_shape, dtype=key_states.dtype, device=key_states.device)
    
    # Set up launch configuration
    BATCH_TILE = 4      # Process 4 batch elements per program
    HEADS_TILE = 8      # Process 8 heads per program
    SEQ_TILE = 64       # Process 64 sequence elements per program
    HEAD_DIM_TILE = 32  # Process 32 head dimension elements per program
    EXPANDS_TILE = 4    # Process 4 expand dimensions per program
    
    # Calculate grid size
    batch_grid = (batch_size + BATCH_TILE - 1) // BATCH_TILE
    heads_grid = (num_heads + HEADS_TILE - 1) // HEADS_TILE
    seq_grid = (seq_len_per_head + SEQ_TILE - 1) // SEQ_TILE
    head_dim_grid = (head_dim + HEAD_DIM_TILE - 1) // HEAD_DIM_TILE
    expands_grid = (num_expands + EXPANDS_TILE - 1) // EXPANDS_TILE
    
    # Launch kernel
    slice_expand_kernel[(batch_grid, heads_grid, seq_grid, head_dim_grid, expands_grid)](
        key_states_ptr=key_states,
        output_ptr=output,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_per_head=seq_len_per_head,
        head_dim=head_dim,
        num_expands=num_expands,
        
        BATCH_TILE=BATCH_TILE,
        HEADS_TILE=HEADS_TILE,
        SEQ_TILE=SEQ_TILE,
        HEAD_DIM_TILE=HEAD_DIM_TILE,
        EXPANDS_TILE=EXPANDS_TILE,
    )
    
    return output

def replacement_func():
    return slice_expand_fused