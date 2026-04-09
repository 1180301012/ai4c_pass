import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern: Fuse embedding with slicing and padding operations for better performance
    """
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch.nn.functional.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    return tmp_7

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_fused_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    seq_len,
    embed_dim,
    vocab_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized kernel that fuses embedding lookup, slicing, and concatenation operations.
    This eliminates multiple memory writes by computing the final concatenated result directly.
    """
    # Get program IDs for 2D grid
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.program_id(2)
    
    if m >= batch_size or n >= seq_len or k >= embed_dim * 3:
        return
    
    # Calculate the combined embedding dimension (3x original)
    total_embed_dim = embed_dim * 3
    
    # Get batch and sequence index
    batch_idx = m
    seq_idx = n
    
    # Load token ID
    token_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx)
    
    # Calculate position in output
    out_pos = batch_idx * seq_len * total_embed_dim + seq_idx * total_embed_dim + k
    out_offset = out_pos
    
    result = tl.zeros(1, dtype=tl.float16)
    
    # Three regions: [right_shifted, original, left_shifted]
    region_offset = k // embed_dim  # 0, 1, or 2
    local_k = k % embed_dim
    
    if region_offset == 0:
        # Right shifted region: contains embedding from next token (with zero at beginning)
        if seq_idx < seq_len - 1:
            next_token_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx + 1)
            if next_token_id < vocab_size and next_token_id >= 0:
                next_offset = next_token_id * embed_dim + local_k
                result = tl.load(weight_ptr + next_offset)
        # For local_k=0 (first element), use 0 (implicit in pattern)
    elif region_offset == 1:
        # Original embedding region
        if token_id < vocab_size and token_id >= 0:
            embed_offset = token_id * embed_dim + local_k
            result = tl.load(weight_ptr + embed_offset)
    elif region_offset == 2:
        # Left shifted region: contains embedding from previous token (with zero at end)
        if seq_idx > 0:
            prev_token_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx - 1)
            if prev_token_id < vocab_size and prev_token_id >= 0:
                prev_offset = prev_token_id * embed_dim + local_k
                result = tl.load(weight_ptr + prev_offset)
        # For local_k = embed_dim-1 (last element), use 0 (implicit in pattern)
    
    # Store the result
    tl.store(output_ptr + out_offset, result)

@torch.fx.wrap
def optimized_fused_operations(input_ids, weight):
    """
    Function that fuses embedding lookup, slicing, and padding operations.
    Returns the concatenated result directly without intermediate allocations.
    """
    batch_size, seq_len = input_ids.shape
    embed_dim = weight.shape[1]
    vocab_size = weight.shape[0]
    
    # Output shape: [batch_size, seq_len, embed_dim * 3]
    total_embed_dim = embed_dim * 3
    output_shape = (batch_size, seq_len, total_embed_dim)
    output = torch.empty(output_shape, dtype=weight.dtype, device=weight.device)
    
    # Grid configurations for 3D parallelism
    grid_m = (batch_size + 7) // 8
    grid_n = (seq_len + 7) // 8
    grid_k = (total_embed_dim + 7) // 8
    
    # Launch 3D grid kernel
    optimized_fused_kernel[(grid_m, grid_n, grid_k)](
        input_ids_ptr=input_ids,
        weight_ptr=weight,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=64,
    )
    
    return output

def replacement_func():
    return optimized_fused_operations