import torch
import triton
import triton.language as tl

def pattern(in_1, in_2, in_3, in_5, in_6, in_7):
    """
    Pattern to match core RoPE operations:
    - Negate in_1
    - Slice in_3 with stride 2
    - Stack, reshape, multiply, add, concatenate, type_as
    This matches models like eva02_small_patch14_224 with shape (1, 6, 256, 64)
    """
    tmp_1 = -in_1
    tmp_2 = in_3[Ellipsis, slice(None, None, 2)]
    tmp_3 = torch.stack([tmp_1, tmp_2], -1)
    tmp_4 = tmp_3.reshape((1, 6, 256, 64))
    tmp_5 = tmp_4 * in_6
    tmp_6 = in_5 + tmp_5
    tmp_7 = torch.cat([in_2, tmp_6], dim=2)
    tmp_8 = tmp_7.type_as(in_7)
    return tmp_8

def replacement_args(in_1, in_2, in_3, in_5, in_6, in_7):
    return (in_1, in_2, in_3, in_5, in_6, in_7)

@triton.jit
def fused_rope_kernel(
    in_1_ptr,  # [batch, heads, seq_len, half_dim]
    in_2_ptr,  # [batch, heads, 1, dim]
    in_3_ptr,  # [batch, heads, seq_len, dim]
    in_5_ptr,  # [batch, heads, seq_len, dim]
    in_6_ptr,  # [seq_len, dim]
    output_ptr,  # [batch, heads, seq_len+1, dim]
    batch: tl.constexpr,
    heads: tl.constexpr,
    seq_len: tl.constexpr,
    half_dim: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that processes:
    1. Copy in_2 to output position 0
    2. Compute RoPE for positions 1 to seq_len+1
    """
    pid = tl.program_id(0)
    
    # Total elements including in_2
    total_elements = batch * heads * (seq_len + 1) * dim
    
    # Compute which elements this program handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decompose linear index to 4D indices
    b_idx = (offsets // (heads * (seq_len + 1) * dim)) % batch
    h_idx = (offsets // ((seq_len + 1) * dim)) % heads
    s_idx = (offsets // dim) % (seq_len + 1)
    d_idx = offsets % dim
    
    # Check if this is position 0 (from in_2) or positions 1+ (from RoPE computation)
    is_first_position = (s_idx == 0)
    
    # For position 0, just copy from in_2
    in_2_offset = b_idx * (heads * dim) + h_idx * dim + d_idx
    val_in_2 = tl.load(in_2_ptr + in_2_offset, mask=mask & is_first_position, other=0.0)
    
    # For positions 1+, compute RoPE
    # Adjust s_idx to be 0-based for indexing into in_1, in_3, in_5, in_6
    s_idx_adjusted = s_idx - 1
    
    # Check if this is an even or odd dimension
    is_even = (d_idx % 2) == 0
    half_d_idx = d_idx // 2
    
    # Compute input offsets
    in_1_offset = b_idx * (heads * seq_len * half_dim) + h_idx * (seq_len * half_dim) + s_idx_adjusted * half_dim + half_d_idx
    in_3_offset = b_idx * (heads * seq_len * dim) + h_idx * (seq_len * dim) + s_idx_adjusted * dim + half_d_idx * 2
    
    # Load from in_1 and in_3
    val_in_1 = tl.load(in_1_ptr + in_1_offset, mask=mask & ~is_first_position, other=0.0)
    val_in_3 = tl.load(in_3_ptr + in_3_offset, mask=mask & ~is_first_position, other=0.0)
    
    # Stack: even positions get -in_1, odd positions get in_3[::2]
    stacked_val = tl.where(is_even, -val_in_1, val_in_3)
    
    # Load in_6 (sin_emb) - shape [seq_len, dim]
    in_6_offset = s_idx_adjusted * dim + d_idx
    val_in_6 = tl.load(in_6_ptr + in_6_offset, mask=mask & ~is_first_position, other=0.0)
    
    # Multiply
    mul_result = stacked_val * val_in_6
    
    # Load in_5 for addition
    in_5_offset = b_idx * (heads * seq_len * dim) + h_idx * (seq_len * dim) + s_idx_adjusted * dim + d_idx
    val_in_5 = tl.load(in_5_ptr + in_5_offset, mask=mask & ~is_first_position, other=0.0)
    
    # Add
    add_result = val_in_5 + mul_result
    
    # Select between in_2 (position 0) and RoPE result (positions 1+)
    final_result = tl.where(is_first_position, val_in_2, add_result)
    
    # Store to output
    tl.store(output_ptr + offsets, final_result, mask=mask)


@torch.fx.wrap
def fused_rope_forward(in_1, in_2, in_3, in_5, in_6, in_7):
    """
    Fused RoPE forward pass
    Returns: tmp_8 (the concatenated and casted result)
    """
    batch, heads, seq_len, half_dim = in_1.shape
    dim = half_dim * 2
    
    # Allocate output for tmp_8 (the concatenated result)
    tmp_8 = torch.empty((batch, heads, seq_len + 1, dim), device=in_1.device, dtype=in_7.dtype)
    
    BLOCK_SIZE = 1024
    
    # Compute everything in one kernel
    total_elements = batch * heads * (seq_len + 1) * dim
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_rope_kernel[grid](
        in_1, in_2, in_3, in_5, in_6, tmp_8,
        batch, heads, seq_len, half_dim, dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return tmp_8

def replacement_func():
    return fused_rope_forward