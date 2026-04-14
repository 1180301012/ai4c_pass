import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch.nn.functional.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim = 2)
    return (tmp_7,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def full_embedding_window_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    embed_dim: tl.constexpr,
    BLOCK_EMBED: tl.constexpr,
):
    # Calculate program id
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    embed_idx = tl.program_id(2)
    
    # Each token position needs 3 * embed_dim output space
    total_embed_dim = embed_dim * 3
    output_offset = (batch_idx * seq_len + seq_idx) * total_embed_dim + embed_idx * BLOCK_EMBED
    
    # Get input id for this batch/seq position
    current_input_idx = batch_idx * seq_len + seq_idx
    current_id = tl.load(input_ids_ptr + current_input_idx)
    
    # Load current token embedding
    current_base = current_id * embed_dim
    current_read_offset = current_base + embed_idx * BLOCK_EMBED
    current_data = tl.load(embedding_weight_ptr + current_read_offset + tl.arange(0, BLOCK_EMBED), 
                          mask=embed_idx * BLOCK_EMBED + tl.arange(0, BLOCK_EMBED) < embed_dim, 
                          other=0.0)
    
    # Initialize output window: [left_context, current, right_context]
    left_context = 0.0
    right_context = 0.0
    
    # Load left context (previous token) if exists
    if seq_idx > 0:
        left_input_idx = current_input_idx - 1
        left_id = tl.load(input_ids_ptr + left_input_idx)
        left_base = left_id * embed_dim
        left_read_offset = left_base + embed_idx * BLOCK_EMBED
        left_context = tl.load(embedding_weight_ptr + left_read_offset + tl.arange(0, BLOCK_EMBED), 
                              mask=embed_idx * BLOCK_EMBED + tl.arange(0, BLOCK_EMBED) < embed_dim, 
                              other=0.0)
    
    # Load right context (next token) if exists
    if seq_idx < seq_len - 1:
        right_input_idx = current_input_idx + 1
        right_id = tl.load(input_ids_ptr + right_input_idx)
        right_base = right_id * embed_dim
        right_read_offset = right_base + embed_idx * BLOCK_EMBED
        right_context = tl.load(embedding_weight_ptr + right_read_offset + tl.arange(0, BLOCK_EMBED), 
                               mask=embed_idx * BLOCK_EMBED + tl.arange(0, BLOCK_EMBED) < embed_dim, 
                               other=0.0)
    
    # Store left context, current, and right context contiguously
    tl.store(output_ptr + output_offset + tl.arange(0, BLOCK_EMBED), left_context,
             mask=(embed_idx * BLOCK_EMBED + tl.arange(0, BLOCK_EMBED)) < embed_dim)
    
    tl.store(output_ptr + output_offset + embed_dim + tl.arange(0, BLOCK_EMBED), current_data,
             mask=(embed_idx * BLOCK_EMBED + tl.arange(0, BLOCK_EMBED)) < embed_dim)
    
    tl.store(output_ptr + output_offset + embed_dim * 2 + tl.arange(0, BLOCK_EMBED), right_context,
             mask=(embed_idx * BLOCK_EMBED + tl.arange(0, BLOCK_EMBED)) < embed_dim)

@torch.fx.wrap
def full_embedding_window_optimized(in_0, in_1):
    batch_size = in_0.shape[0]
    seq_len = in_0.shape[1]
    embed_dim = in_1.shape[1]
    
    # Output has 3x the embedding dimension (left context, current, right context)
    total_embed_dim = embed_dim * 3
    output_shape = (batch_size, seq_len, total_embed_dim)
    output = torch.zeros(output_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Grid configuration
    BLOCK_EMBED = 64  # Optimal for most GPU architectures
    num_batches = (batch_size + 1 - 1) // 1
    num_seqs = (seq_len + 1 - 1) // 1
    num_embeds = (embed_dim + BLOCK_EMBED - 1) // BLOCK_EMBED
    
    full_embedding_window_kernel[(num_batches, num_seqs, num_embeds)](
        in_0,
        in_1,
        output,
        batch_size,
        seq_len,
        embed_dim,
        BLOCK_EMBED,
    )
    
    return output

def replacement_func():
    return full_embedding_window_optimized