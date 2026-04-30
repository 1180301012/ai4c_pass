import torch
import triton
import triton.language as tl

def pattern(in_4, embedding_weights, in_0):
    """
    Pattern to match: position embedding lookup + element-wise addition
    in_4 is positions tensor
    embedding_weights are the position embeddings
    in_0 is the input to add to
    """
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, embedding_weights, None, None, 2.0, False, False)
    tmp_12 = tmp_11.to(embedding_weights.device)
    tmp_13 = in_0 + tmp_12
    return tmp_13

def replacement_args(in_4, embedding_weights, in_0):
    return (in_4, embedding_weights, in_0)

@triton.jit
def embedding_add_kernel(
    positions_ptr, embedding_ptr, input_ptr, output_ptr,
    seq_len, batch_size, hidden_size,
    embedding_rows, BLOCK_SIZE: tl.constexpr
):
    # Get batch and position indices
    batch_pid = tl.program_id(0)
    pos_pid = tl.program_id(1)
    
    # Calculate output offset
    output_offset = (batch_pid * seq_len + pos_pid) * hidden_size
    
    # Calculate position index (positions[pos_pid] + 2)
    pos_val = tl.load(positions_ptr + pos_pid).to(tl.int32) + 2
    
    # Check if position is valid and not padding_idx (2.0 means padding_idx=2 in PyTorch)
    # In embedding with padding_idx=2, positions with index 2 get zero vectors
    is_valid_pos = pos_val < embedding_rows
    
    # Load the embedding vector for this position
    emb_offsets = pos_val * hidden_size + tl.arange(0, BLOCK_SIZE)
    emb_mask = emb_offsets < embedding_rows * hidden_size
    
    # Load embedding (bfloat16/float16) and convert to float32 for computation
    emb_vec = tl.load(embedding_ptr + emb_offsets, mask=emb_mask, other=0.0).to(tl.float32)
    
    # Zero out embedding for invalid positions (padding_idx behavior)
    emb_vec = tl.where(is_valid_pos, emb_vec, 0.0)
    
    # Load input tensor
    input_offsets = output_offset + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offsets < batch_size * seq_len * hidden_size
    
    input_vec = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
    
    # Add embedding to input
    output_vec = input_vec + emb_vec
    
    # Store result
    tl.store(output_ptr + input_offsets, output_vec, mask=input_mask)

@torch.fx.wrap
def fused_embedding_add(in_4, embedding_weights, in_0):
    """
    Fused embedding lookup + addition kernel.
    Combines unsqueeze(0), +2, embedding, to(device), + in_0
    """
    # Get dimensions
    batch_size, seq_len, hidden_size = in_0.shape
    embedding_rows = embedding_weights.shape[0]
    
    # Output tensor
    output = torch.empty_like(in_0)
    
    # Block size for hidden dimension
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    # Grid: (batch_size, seq_len)
    grid = (batch_size, seq_len)
    
    embedding_add_kernel[grid](
        positions_ptr=in_4,
        embedding_ptr=embedding_weights,
        input_ptr=in_0,
        output_ptr=output,
        seq_len=seq_len,
        batch_size=batch_size,
        hidden_size=hidden_size,
        embedding_rows=embedding_rows,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_embedding_add