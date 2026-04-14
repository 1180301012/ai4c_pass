import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def embedding_windowed_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    vocab_size: tl.constexpr,
    embed_dim: tl.constexpr,
    EMBEDDING_DIM_PER_HEAD: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_EMBED: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    embed_idx = tl.program_id(2)
    
    # Calculate base offset for this program
    batch_offset = batch_idx * seq_len * embed_dim * 3
    seq_offset = seq_idx * embed_dim * 3
    embed_offset = embed_idx * BLOCK_EMBED
    
    read_ptr = embedding_weight_ptr + input_ids_ptr[batch_idx * seq_len + seq_idx] * embed_dim + embed_offset
    
    # Read the original embedding
    orig = tl.load(read_ptr + tl.arange(0, BLOCK_EMBED), mask=embed_idx + tl.arange(0, BLOCK_EMBED) < embed_dim, other=0.0)
    
    # For the windowed output: [left_context, current, right_context]
    # Each has embed_dim, so total is embed_dim * 3
    base_offset = batch_offset + seq_offset + embed_offset
    
    # Left context: previous token embedding (or zero for first token)
    if seq_idx > 0:
        left_ptr = embedding_weight_ptr + input_ids_ptr[batch_idx * seq_len + seq_idx - 1] * embed_dim + embed_offset
        left = tl.load(left_ptr + tl.arange(0, BLOCK_EMBED), mask=embed_idx + tl.arange(0, BLOCK_EMBED) < embed_dim, other=0.0)
    else:
        left = 0.0
    
    # Current token embedding
    current = orig
    
    # Right context: next token embedding (or zero for last token)  
    if seq_idx < seq_len - 1:
        right_ptr = embedding_weight_ptr + input_ids_ptr[batch_idx * seq_len + seq_idx + 1] * embed_dim + embed_offset
        right = tl.load(right_ptr + tl.arange(0, BLOCK_EMBED), mask=embed_idx + tl.arange(0, BLOCK_EMBED) < embed_dim, other=0.0)
    else:
        right = 0.0
    
    # Store in output: [left_context[embed_dim], current[embed_dim], right_context[embed_dim]]
    output_base = output_ptr + base_offset
    
    # Store left context
    tl.store(output_base + tl.arange(0, BLOCK_EMBED), left, mask=embed_idx + tl.arange(0, BLOCK_EMBED) < embed_dim)
    
    # Store current (orig at offset embed_dim)
    current_offset = output_base + embed_dim
    tl.store(current_offset + tl.arange(0, BLOCK_EMBED), current, mask=embed_idx + tl.arange(0, BLOCK_EMBED) < embed_dim)
    
    # Store right context (at offset embed_dim * 2)
    right_offset = output_base + embed_dim * 2
    tl.store(right_offset + tl.arange(0, BLOCK_EMBED), right, mask=embed_idx + tl.arange(0, BLOCK_EMBED) < embed_dim)

@torch.fx.wrap  
def optimized_embedding_windowed(in_0, in_1):
    batch_size = in_0.shape[0]
    seq_len = in_0.shape[1]
    embed_dim = in_1.shape[1]
    
    output_shape = (batch_size, seq_len, embed_dim * 3)
    output = torch.zeros(output_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Get input ids as a contiguous 1D tensor for easier indexing
    input_ids_flat = in_0.contiguous().view(-1)
    
    # Grid configuration
    EMBEDDING_DIM_PER_HEAD = 64  # Optimal for most GPU architectures
    BLOCK_BATCH = 1
    BLOCK_SEQ = 1
    BLOCK_EMBED = min(EMBEDDING_DIM_PER_HEAD, embed_dim)
    
    num_batches = (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH
    num_seqs = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    num_embeds = (embed_dim + BLOCK_EMBED - 1) // BLOCK_EMBED
    
    embedding_windowed_kernel[
        (num_batches, num_seqs, num_embeds),
        (BLOCK_BATCH, BLOCK_SEQ, BLOCK_EMBED),
    ](
        input_ids_flat,
        in_1,
        output,
        batch_size,
        seq_len,
        in_1.shape[0],
        embed_dim,
        EMBEDDING_DIM_PER_HEAD,
        BLOCK_BATCH,
        BLOCK_SEQ,
        BLOCK_EMBED,
    )
    
    return output

def replacement_func():
    return optimized_embedding_windowed