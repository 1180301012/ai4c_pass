import torch
import triton
import triton.language as tl

# Optimized embedding lookup kernel with autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def embedding_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    N,  # embedding dimension (1536)
    stride_indices_batch,  # stride of indices in batch dimension
    stride_indices_seq,    # stride of indices in sequence dimension
    num_indices,           # total number of indices to lookup
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel for embedding lookup.
    
    Args:
        indices_ptr: Pointer to indices tensor [batch, seq_len]
        weight_ptr: Pointer to embedding weight [vocab_size, embed_dim]
        output_ptr: Pointer to output tensor [batch, seq_len, embed_dim]
        N: embedding dimension (1536)
        stride_indices_batch: stride of indices tensor in batch dim
        stride_indices_seq: stride of indices tensor in seq dim
        num_indices: total number of indices (batch * seq_len)
    """
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # Calculate number of blocks in M dimension (indices)
    num_pid_m = tl.num_programs(1)
    
    # Offsets for indices
    indices_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_indices = indices_offsets < num_indices
    
    # Load indices (flattened view)
    indices = tl.load(indices_ptr + indices_offsets, mask=mask_indices, other=0)
    
    # Calculate output offsets
    # output[batch, seq, embed_dim] -> flattened as [batch * seq, embed_dim]
    batch_id = pid_batch
    seq_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Prepare output pointers for each embed_dim block
    output_base = pid_batch * num_indices * N + pid_m * BLOCK_SIZE_M * N
    
    # Process embed_dim in blocks
    for pid_n in range(tl.cdiv(N, BLOCK_SIZE_N)):
        n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = n_offsets < N
        
        # Weight indices: [indices, n_offsets]
        # weight[stride, :] - weight is [vocab_size, embed_dim] 
        # weight[row, col] -> weight[row * N + col]
        weight_row_offsets = indices * N + n_offsets
        mask_weight = mask_indices & mask_n
        
        # Load embedding vectors
        weight_vals = tl.load(weight_ptr + weight_row_offsets, mask=mask_weight, other=0.0)
        
        # Store output
        out_offsets = output_base + tl.arange(0, BLOCK_SIZE_M)[:, None] * N + n_offsets[None, :]
        out_mask = mask_indices[:, None] & mask_n[None, :]
        tl.store(output_ptr + out_offsets, weight_vals, mask=out_mask)


def pattern(in_0, in_1, in_2):
    """
    Match the embedding lookup pattern.
    
    Pattern matches:
    - torch.nn.functional.embedding with: padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False
    - .long() conversion on attention mask
    
    Returns:
        - tmp_3: embedding output [batch, seq_len, embed_dim]
        - tmp_4: attention mask converted to long
    """
    tmp_3 = torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False)
    tmp_4 = in_0.long()
    return tmp_3, tmp_4


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement function.
    
    in_0: attention_mask tensor (returned as-is with .long() conversion)
    in_1: input_ids tensor for embedding lookup
    in_2: embedding weight table [vocab_size, embed_dim]
    """
    return (in_0, in_1, in_2, "embedding_lookup")


def triton_embedding_lookup(attention_mask, input_ids, weight, route=""):
    """
    Custom Triton-based embedding lookup.
    
    Args:
        attention_mask: Tensor [batch, seq_len] - attention mask (returned as attention_mask.long())
        input_ids: Tensor [batch, seq_len] - indices for embedding lookup
        weight: Tensor [vocab_size, embed_dim] - embedding weight table
        route: Routing string for shared replacement_func
    
    Returns:
        Tuple of (embedding_output, attention_mask_long)
    """
    if route != "embedding_lookup":
        # This route shouldn't be called directly from this pass
        # Return a placeholder (actual computation handled elsewhere)
        return attention_mask.long(), torch.zeros_like(attention_mask).long()
    
    batch_size, seq_len = input_ids.shape
    vocab_size, embed_dim = weight.shape
    num_indices = batch_size * seq_len
    
    # Allocate output tensor
    output = torch.empty((batch_size, seq_len, embed_dim), 
                         dtype=weight.dtype, 
                         device=weight.device)
    
    # For smaller batch sizes, use fewer programs
    # Grid: (batch_size, num_index_blocks)
    # Heuristic: one program per batch item
    grid = (batch_size,)
    
    embedding_kernel[grid](
        indices_ptr=input_ids,
        weight_ptr=weight,
        output_ptr=output,
        N=embed_dim,
        stride_indices_batch=seq_len,
        stride_indices_seq=1,
        num_indices=num_indices,
    )
    
    return output, attention_mask.long()


# Wrap the Triton kernel for FX
@torch.fx.wrap
def triton_embedding_lookup_wrapper(attention_mask, input_ids, weight, route=""):
    return triton_embedding_lookup(attention_mask, input_ids, weight, route)


def replacement_func():
    """
    Returns the replacement function for the embedding lookup pattern.
    """
    return triton_embedding_lookup_wrapper