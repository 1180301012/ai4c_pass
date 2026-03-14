import torch
import triton
import triton.language as tl

def pattern(weight, hidden_states, key_states):
    # Exact pattern matching from graph computation
    tmp_1 = torch.nn.functional.linear(hidden_states, weight, None)
    tmp_2 = tmp_1.view((4, 512, -1, 128))
    tmp_3 = tmp_2.transpose(1, 2)
    
    return tmp_3

def replacement_args(weight, hidden_states, key_states):
    return (weight, hidden_states, key_states)

@triton.jit
def linear_view_transpose_kernel(
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
    # Calculate program IDs
    pid_batch = tl.cdiv(batch_size, BATCH_TILE)
    pid_seq = tl.cdiv(seq_len, SEQUENCE_TILE)
    pid_hidden = tl.cdiv(hidden_size // 128, HIDDEN_SIZE_TILE)
    
    # Parallelize over batch, sequence, and the split dimension
    for bid in range(tl.program_id(0)):
        if bid >= pid_batch:
            break
        for sid in range(tl.program_id(1)):
            if sid >= pid_seq:
                break
        for hid in range(tl.program_id(2)):
            if hid >= pid_hidden:
                break
            
            # Calculate batch, sequence, and hidden indices
            b = bid * BATCH_TILE + tl.arange(0, BATCH_TILE)
            s = sid * SEQUENCE_TILE + tl.arange(0, SEQUENCE_TILE)
            h_div_128 = hid * HIDDEN_SIZE_TILE + tl.arange(0, HIDDEN_SIZE_TILE)
            h_in_chunk = h_div_128 * 128 + tl.arange(0, 128)
            
            # Create masks for bounds checking
            b_mask = b < batch_size
            s_mask = s < seq_len
            h_div_128_mask = h_div_128 < (hidden_size // 128)
            h_in_chunk_mask = h_in_chunk < hidden_size
            
            # Load weight matrix column for this hidden chunk
            # weight shape: [hidden_size, 2048]
            weight_start = h_in_chunk[:, None] * 2048 + tl.arange(0, 2048)[None, :]
            weight_mask = h_in_chunk_mask[:, None] & (tl.arange(0, 2048)[None, :] < 2048)
            weight_chunk = tl.load(weight_ptr + weight_start, mask=weight_mask, other=0.0)
            
            # Load hidden states for this batch/sequence
            # hidden_states shape: [batch_size, seq_len, 2048]
            hs_start = b[:, None, None] * seq_len * 2048 + s[None, :, None] * 2048 + tl.arange(0, 2048)[None, None, :]
            hs_mask = b_mask[:, None, None] & s_mask[None, :, None] & (tl.arange(0, 2048)[None, None, :] < 2048)
            hs_chunk = tl.load(hidden_states_ptr + hs_start, mask=hs_mask, other=0.0)
            
            # Compute matrix multiplication: [hidden_chunk_size, 2048] @ [2048] -> [hidden_chunk_size]
            # For each (b, s) pair, compute dot product with weight chunk
            linear_result = tl.sum(weight_chunk * hs_chunk, axis=2)
            
            # Transpose the results: [batch_size, seq_len, hidden_size//128, 128] -> [batch_size, hidden_size//128, seq_len, 128]
            # Store back to output with proper transposition
            output_indices = b[:, None, None, None] * (hidden_size // 128) * seq_len * 128 + \
                           h_div_128[None, :, None, None] * seq_len * 128 + \
                           s[None, None, :, None] * 128 + \
                           tl.arange(0, 128)[None, None, None, :]
            
            output_mask = b_mask[:, None, None, None] & h_div_128_mask[None, :, None, None] & s_mask[None, None, :, None] & h_in_chunk_mask[None, None, None, None]
            
            # Store the transposed result
            tl.store(output_ptr + output_indices, linear_result, mask=output_mask)

@torch.fx.wrap
def linear_view_transpose_fused(weight, hidden_states, key_states):
    # Get tensor shapes
    batch_size, seq_len, hidden_size = hidden_states.shape
    _, weight_hidden_size = weight.shape
    
    # Output shape: [batch_size, hidden_size//128, seq_len, 128]
    out_shape = (batch_size, hidden_size // 128, seq_len, 128)
    output = torch.empty(out_shape, dtype=hidden_states.dtype, device=hidden_states.device)
    
    # Set up launch configuration
    HIDDEN_SIZE_TILE = 4  # Process 4 chunks of 128 elements each
    SEQUENCE_TILE = 64    # Process 64 sequence elements per program
    BATCH_TILE = 8        # Process 8 batch elements per program
    
    # Calculate grid size
    batch_grid = (batch_size + BATCH_TILE - 1) // BATCH_TILE
    seq_grid = (seq_len + SEQUENCE_TILE - 1) // SEQUENCE_TILE
    hidden_grid = (hidden_size // 128 + HIDDEN_SIZE_TILE - 1) // HIDDEN_SIZE_TILE
    
    # Launch kernel
    linear_view_transpose_kernel[(batch_grid, seq_grid, hidden_grid)](
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
    return linear_view_transpose_fused