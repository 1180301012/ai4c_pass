import torch
import triton
import triton.language as tl

def pattern(weight, hidden_states, key_states):
    # Exact pattern matching for Graph 5
    tmp_1 = torch.nn.functional.linear(hidden_states, weight, None)
    tmp_2 = tmp_1.view((4, 512, -1, 128))
    tmp_3 = tmp_2.transpose(1, 2)
    
    return tmp_3

def replacement_args(weight, hidden_states, key_states):
    return (weight, hidden_states, key_states)

@triton.jit
def linear_view_transpose_kernel_graph5(
    weight_ptr,
    hidden_states_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_size,
    out_size_3,
    HIDDEN_SIZE_TILE: tl.constexpr,
    SEQUENCE_TILE: tl.constexpr,
    BATCH_TILE: tl.constexpr,
):
    # Program IDs for 3D grid
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_hidden = tl.program_id(2)
    
    # Calculate indices
    batch_idx = pid_batch * BATCH_TILE + tl.arange(0, BATCH_TILE)
    seq_idx = pid_seq * SEQUENCE_TILE + tl.arange(0, SEQUENCE_TILE)
    
    # For hidden dimension, we need to compute in chunks of 128
    hidden_chunk_idx = pid_hidden * HIDDEN_SIZE_TILE + tl.arange(0, HIDDEN_SIZE_TILE)
    hidden_idx = hidden_chunk_idx * 128 + tl.arange(0, 128)
    
    # Masks
    batch_mask = batch_idx < batch_size
    seq_mask = seq_idx < seq_len
    hidden_chunk_mask = hidden_chunk_idx < (hidden_size // 128)
    hidden_mask = hidden_idx < hidden_size
    
    # Load weight chunk
    weight_start = hidden_idx[:, None] * 2048 + tl.arange(0, 2048)[None, :]
    weight_mask = hidden_mask[:, None] & (tl.arange(0, 2048)[None, :] < 2048)
    weight_chunk = tl.load(weight_ptr + weight_start, mask=weight_mask, other=0.0)
    
    # Load hidden states
    hs_start = batch_idx[:, None, None] * seq_len * 2048 + seq_idx[None, :, None] * 2048 + tl.arange(0, 2048)[None, None, :]
    hs_mask = batch_mask[:, None, None] & seq_mask[None, :, None] & (tl.arange(0, 2048)[None, None, :] < 2048)
    hs_chunk = tl.load(hidden_states_ptr + hs_start, mask=hs_mask, other=0.0)
    
    # Compute linear transformation
    linear_result = tl.sum(weight_chunk * hs_chunk, axis=2)
    
    # Store transposed result: [batch_size, hidden_size//128, seq_len, 128]
    output_idx = batch_idx[:, None, None, None] * (hidden_size // 128) * seq_len * 128 + \
                 hidden_chunk_idx[None, :, None, None] * seq_len * 128 + \
                 seq_idx[None, None, :, None] * 128 + \
                 tl.arange(0, 128)[None, None, None, :]
    
    output_mask = batch_mask[:, None, None, None] & hidden_chunk_mask[None, :, None, None] & seq_mask[None, None, :, None] & hidden_mask[None, None, None, None]
    
    tl.store(output_ptr + output_idx, linear_result, mask=output_mask)

@torch.fx.wrap
def linear_view_transpose_fused_graph5(weight, hidden_states, key_states):
    # Get tensor shapes for Graph 5
    batch_size, seq_len, hidden_size = hidden_states.shape
    
    # Output shape: [batch_size, hidden_size//128, seq_len, 128]
    out_shape = (batch_size, hidden_size // 128, seq_len, 128)
    output = torch.empty(out_shape, dtype=hidden_states.dtype, device=hidden_states.device)
    
    # Tile sizes
    BATCH_TILE = 4
    SEQUENCE_TILE = 64
    HIDDEN_SIZE_TILE = 4
    
    # Grid size
    batch_grid = (batch_size + BATCH_TILE - 1) // BATCH_TILE
    seq_grid = (seq_len + SEQUENCE_TILE - 1) // SEQUENCE_TILE
    hidden_grid = (hidden_size // 128 + HIDDEN_SIZE_TILE - 1) // HIDDEN_SIZE_TILE
    
    # Launch kernel
    linear_view_transpose_kernel_graph5[(batch_grid, seq_grid, hidden_grid)](
        weight_ptr=weight,
        hidden_states_ptr=hidden_states,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        out_size_3=128,
        HIDDEN_SIZE_TILE=HIDDEN_SIZE_TILE,
        SEQUENCE_TILE=SEQUENCE_TILE,
        BATCH_TILE=BATCH_TILE,
    )
    
    return output

def replacement_func():
    return linear_view_transpose_fused_graph5