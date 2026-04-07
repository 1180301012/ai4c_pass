import torch
import triton
import triton.language as tl

def pattern(input_ids, weight):
    # Match the exact computation pattern from model.py
    tmp_2 = torch.nn.functional.embedding(input_ids, weight, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch.nn.functional.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    return tmp_7

def replacement_args(input_ids, weight):
    return (input_ids, weight)

@triton.jit
def embedding_sliding_window_kernel(
    input_ids_ptr, 
    weight_ptr, 
    output_ptr,
    seq_len, 
    embed_dim,
    vocab_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program id for matrix tiling
    m = tl.program_id(0)
    n_offset = tl.program_id(1) * BLOCK_SIZE_N
    
    # Create mask for this program
    mask_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < seq_len
    mask_n = n_offset + tl.arange(0, BLOCK_SIZE_N) < embed_dim
    
    # Load input token ids (only need the ones that are valid)
    input_ids = tl.load(input_ids_ptr + m * BLOCK_SIZE_M, mask=mask_m, other=0).to(tl.int32)
    
    # Load weight matrix columns for the embed_dim range
    weight_offset = tl.arange(0, BLOCK_SIZE_N) 
    weight_vals = tl.load(weight_ptr + input_ids[:, None] * embed_dim + (n_offset + weight_offset[None, :]), 
                         mask=(input_ids[:, None] < vocab_size) & (mask_n[None, :]), 
                         other=0.0).to(tl.float32)
    
    # Apply the scaling factor (2.0)
    weight_vals *= 2.0
    
    # Compute sliding window output directly into final 3-channels
    # Output shape: [BLOCK_SIZE_M, BLOCK_SIZE_N, 3]
    # Channel 0: left context (embedding from next token, padded with 0 at start)
    # Channel 1: center context (current token embedding)  
    # Channel 2: right context (embedding from previous token, padded with 0 at end)
    
    # Initialize output channels  
    left_context = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    center_context = weight_vals
    right_context = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute left context: embedding from next token, handling boundary
    valid_left = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < seq_len - 1
    if BLOCK_SIZE_M > 1:
        # Shift embeddings to the left (next token becomes current)
        next_input_ids = input_ids[1:] if BLOCK_SIZE_M > 1 else input_ids
        next_weight_vals = tl.load(weight_ptr + next_input_ids[:, None] * embed_dim + (n_offset + weight_offset[None, :]), 
                                  mask=(next_input_ids[:, None] < vocab_size) & (mask_n[None, :]) & valid_left[:-1], 
                                  other=0.0).to(tl.float32) * 2.0
        left_context[1:] = next_weight_vals
    else:
        next_input_id = input_ids[0] if seq_len > 1 else 0
        if seq_len > 1 and next_input_id < vocab_size:
            next_weight_val = tl.load(weight_ptr + next_input_id * embed_dim + n_offset, 
                                      mask=mask_n, 
                                      other=0.0).to(tl.float32) * 2.0
            left_context[0] = next_weight_val
    
    # Compute right context: embedding from previous token, handling boundary
    valid_right = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) > 0
    if BLOCK_SIZE_M > 1:
        # Shift embeddings to the right (previous token becomes current)
        prev_input_ids = input_ids[:-1] if BLOCK_SIZE_M > 1 else input_ids
        prev_weight_vals = tl.load(weight_ptr + prev_input_ids[:, None] * embed_dim + (n_offset + weight_offset[None, :]), 
                                  mask=(prev_input_ids[:, None] < vocab_size) & (mask_n[None, :]) & valid_right[1:], 
                                  other=0.0).to(tl.float32) * 2.0
        right_context[:-1] = prev_weight_vals
    else:
        prev_input_id = input_ids[0] if m > 0 else 0
        if m > 0 and prev_input_id < vocab_size:
            prev_weight_val = tl.load(weight_ptr + prev_input_id * embed_dim + n_offset, 
                                      mask=mask_n, 
                                      other=0.0).to(tl.float32) * 2.0
            right_context[0] = prev_weight_val
    
    # Determine actual output sizes based on sequence length bounds
    output_m = min(BLOCK_SIZE_M, seq_len - m * BLOCK_SIZE_M)
    output_n = min(BLOCK_SIZE_N, embed_dim - n_offset)
    
    # Interleave the three channels into the output layout
    # Output layout: [output_m, output_n, 3]
    for i in range(output_m):
        for j in range(output_n):
            base_idx = ((m * BLOCK_SIZE_M + i) * embed_dim * 3 + (n_offset + j) * 3)
            left_data = left_context[i, j]
            center_data = center_context[i, j]
            right_data = right_context[i, j]
            tl.store(output_ptr + base_idx, left_data)
            tl.store(output_ptr + base_idx + 1, center_data)
            tl.store(output_ptr + base_idx + 2, right_data)

@torch.fx.wrap
def optimized_embedding_sliding_window(input_ids, weight):
    # Get dimensions
    seq_len = input_ids.shape[0]
    embed_dim = weight.shape[1]
    vocab_size = weight.shape[0]
    
    # Output shape: [seq_len, embed_dim, 3]
    output_shape = (seq_len, embed_dim, 3)
    output = torch.empty(output_shape, dtype=weight.dtype, device=weight.device)
    
    # Determine block sizes based on typical GPU architecture
    BLOCK_SIZE_M = 64  # Sequence dimension tiling
    BLOCK_SIZE_N = 128  # Embedding dimension tiling
    
    # Calculate grid dimensions
    num_programs_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (embed_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch the kernel
    embedding_sliding_window_kernel[(num_programs_m, num_programs_n)](
        input_ids_ptr=input_ids,
        weight_ptr=weight,
        output_ptr=output,
        seq_len=seq_len,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    return optimized_embedding_sliding_window