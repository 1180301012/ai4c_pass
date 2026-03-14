import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=4),
    ],
    key=['seq_len', 'embed_dim'],
)
@triton.jit
def embedding_kernel(
    input_ids_ptr,
    embedding_table_ptr,
    output_ptr,
    batch_size,
    seq_len,
    embed_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized Triton kernel for embedding lookup.
    Each program processes a subset of the batch and sequence dimensions.
    """
    # Calculate program ID
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # Calculate how many elements this program handles
    total_tokens = batch_size * seq_len
    tokens_per_program = (total_tokens + num_programs - 1) // num_programs
    start_token = pid * tokens_per_program
    end_token = min(start_token + tokens_per_program, total_tokens)
    
    # Process tokens assigned to this program
    for token_idx in range(start_token, end_token):
        batch_idx = token_idx // seq_len
        seq_idx = token_idx % seq_len
        
        # Load the token ID
        input_offset = batch_idx * seq_len + seq_idx
        token_id = tl.load(input_ids_ptr + input_offset).to(tl.int64)
        
        # Calculate the offset in the embedding table for this token
        # embedding_table is [num_tokens, embed_dim], row-major
        embed_offset = token_id * embed_dim
        
        # Load the embedding vector and store to output
        # Output shape: [batch_size, seq_len, embed_dim]
        output_offset = batch_idx * seq_len * embed_dim + seq_idx * embed_dim
        
        # Process embedding dimension in blocks
        for embed_idx in range(0, embed_dim, BLOCK_SIZE_N):
            # Mask for avoiding out-of-bounds
            mask = embed_idx + tl.arange(0, BLOCK_SIZE_N) < embed_dim
            
            # Load from embedding table
            embed_offsets = embed_offset + embed_idx + tl.arange(0, BLOCK_SIZE_N)
            embedding_vals = tl.load(embedding_table_ptr + embed_offsets, mask=mask, other=0.0)
            
            # Store to output
            out_offsets = output_offset + embed_idx + tl.arange(0, BLOCK_SIZE_N)
            tl.store(output_ptr + out_offsets, embedding_vals, mask=mask)


@torch.fx.wrap
def triton_embedding(input_ids, embedding_table):
    """
    Wrapper function for the Triton embedding kernel.
    """
    batch_size, seq_len = input_ids.shape
    embed_dim = embedding_table.shape[1]
    
    # Output shape: [batch_size, seq_len, embed_dim]
    output = torch.empty((batch_size, seq_len, embed_dim), 
                         dtype=embedding_table.dtype, 
                         device=embedding_table.device)
    
    # Calculate grid size
    # Use a reasonable number of programs for parallelism
    # Fixed value that works well for most GPUs
    num_programs = min(80, batch_size * seq_len)
    
    # Launch kernel
    embedding_kernel[(num_programs,)](
        input_ids_ptr=input_ids,
        embedding_table_ptr=embedding_table,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
    )
    
    return output


def pattern(in_0, in_1, in_2):
    """
    Match both the embedding lookup AND the type conversion from the model.
    
    This ensures all 3 inputs are used in the pattern, avoiding dead code.
    The original computation:
    - tmp_3 = embedding(tmp_1, tmp_2, ...)
    - tmp_4 = tmp_0.long()
    - return (tmp_3, tmp_4)
    
    We match both operations to use all inputs and return both outputs.
    """
    # The embedding computation
    embed_output = torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False)
    
    # The type conversion on attention_mask
    long_output = in_0.long()
    
    # Return both outputs - this ensures both are used
    return embed_output, long_output


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement.
    We need all three inputs: in_0 (attention_mask), in_1 (input_ids), in_2 (embedding table).
    """
    return (in_0, in_1, in_2)


def replacement_func():
    """
    Return the replacement function that handles both embedding and type conversion.
    """
    def replacement_impl(attention_mask, input_ids, embedding_table):
        # Use Triton kernel for embedding lookup
        embed_output = triton_embedding(input_ids, embedding_table)
        
        # Simple type conversion for attention_mask
        long_output = attention_mask.long()
        
        return embed_output, long_output
    
    return replacement_impl