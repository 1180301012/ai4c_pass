import torch
import triton
import triton.language as tl

def pattern(key_states, linear_output):
    # Exact pattern matching for Graph 5 slice+expand
    tmp_4 = key_states[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_5 = tmp_4.expand((4, 4, 4, 512, 128))
    
    return tmp_5

def replacement_args(key_states, linear_output):
    return (key_states, linear_output)

@triton.jit
def slice_expand_kernel_graph5(
    key_states_ptr,
    output_ptr,
    batch_size,
    num_heads,
    seq_len_per_head,
    head_dim,
    
    BATCH_TILE: tl.constexpr,
    HEADS_TILE: tl.constexpr,
    SEQ_TILE: tl.constexpr,
    HEAD_DIM_TILE: tl.constexpr,
    EXPANDS_TILE: tl.constexpr,
):
    # Program IDs for 4D grid (batch, heads, seq, head_dim)
    # We broadcast over the expand dimension
    pid_batch = tl.program_id(0)
    pid_heads = tl.program_id(1)
    pid_seq = tl.program_id(2)
    pid_head_dim = tl.program_id(3)
    
    # Calculate indices for input (4D) and output (5D)
    batch_idx = pid_batch * BATCH_TILE + tl.arange(0, BATCH_TILE)
    head_idx = pid_heads * HEADS_TILE + tl.arange(0, HEADS_TILE)
    seq_idx = pid_seq * SEQ_TILE + tl.arange(0, SEQ_TILE)
    head_dim_idx = pid_head_dim * HEAD_DIM_TILE + tl.arange(0, HEAD_DIM_TILE)
    
    # Masks
    batch_mask = batch_idx < batch_size
    head_mask = head_idx < num_heads
    seq_mask = seq_idx < seq_len_per_head
    head_dim_mask = head_dim_idx < head_dim
    
    # Load from 4D key_states: [batch_size, num_heads, seq_len_per_head, head_dim]
    input_start = batch_idx[:, None, None, None] * num_heads * seq_len_per_head * head_dim + \
                 head_idx[None, :, None, None] * seq_len_per_head * head_dim + \
                 seq_idx[None, None, :, None] * head_dim + \
                 head_dim_idx[None, None, None, :]
    
    input_mask = batch_mask[:, None, None, None] & head_mask[None, :, None, None] & seq_mask[None, None, :, None] & head_dim_mask[None, None, None, :]
    
    input_data = tl.load(key_states_ptr + input_start, mask=input_mask, other=0.0)
    
    # Broadcast to 5D output: [batch_size, num_heads, 4, seq_len_per_head, head_dim]
    # For each expand dimension (4), we copy the same data
    output_start = batch_idx[:, None, None, None, None] * num_heads * 4 * seq_len_per_head * head_dim + \
                  head_idx[None, :, None, None, None] * 4 * seq_len_per_head * head_dim + \
                  tl.arange(0, 4)[None, None, :, None, None] * seq_len_per_head * head_dim + \
                  seq_idx[None, None, None, :, None] * head_dim + \
                  head_dim_idx[None, None, None, None, :]
    
    output_mask = batch_mask[:, None, None, None, None] & head_mask[None, :, None, None, None] & \
                  seq_mask[None, None, None, :, None] & head_dim_mask[None, None, None, None, None]
    
    # Broadcast the same input_data for all 4 expand dimensions
    input_expanded = input_data[:, :, None, :, :]  # Add expand dimension
    input_reshaped = input_expanded.reshape(-1)  # Flatten for broadcasting
    
    # Reshape output indices for broadcasting
    flat_output = output_start.reshape(-1)
    flat_output_mask = output_mask.reshape(-1)
    
    # Broadcast the data
    tl.store(output_ptr + flat_output, input_reshaped, mask=flat_output_mask)

@torch.fx.wrap
def slice_expand_fused_graph5(key_states, linear_output):
    # Get key_states shape for Graph 5: [4, 4, 512, 128]
    batch_size, num_heads, seq_len_per_head, head_dim = key_states.shape
    
    # Output shape: [batch_size, num_heads, 4, seq_len_per_head, head_dim]
    expand_shape = (batch_size, num_heads, 4, seq_len_per_head, head_dim)
    output = torch.empty(expand_shape, dtype=key_states.dtype, device=key_states.device)
    
    # Tile sizes
    BATCH_TILE = 2
    HEADS_TILE = 4
    SEQ_TILE = 256
    HEAD_DIM_TILE = 64
    EXPANDS_TILE = 4  # Process all 4 expand dimensions
    
    # Grid size
    batch_grid = (batch_size + BATCH_TILE - 1) // BATCH_TILE
    heads_grid = (num_heads + HEADS_TILE - 1) // HEADS_TILE
    seq_grid = (seq_len_per_head + SEQ_TILE - 1) // SEQ_TILE
    head_dim_grid = (head_dim + HEAD_DIM_TILE - 1) // HEAD_DIM_TILE
    
    # Launch kernel
    slice_expand_kernel_graph5[(batch_grid, heads_grid, seq_grid, head_dim_grid)](
        key_states_ptr=key_states,
        output_ptr=output,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_per_head=seq_len_per_head,
        head_dim=head_dim,
        
        BATCH_TILE=BATCH_TILE,
        HEADS_TILE=HEADS_TILE,
        SEQ_TILE=SEQ_TILE,
        HEAD_DIM_TILE=HEAD_DIM_TILE,
        EXPANDS_TILE=EXPANDS_TILE,
    )
    
    return output

def replacement_func():
    return slice_expand_fused_graph5