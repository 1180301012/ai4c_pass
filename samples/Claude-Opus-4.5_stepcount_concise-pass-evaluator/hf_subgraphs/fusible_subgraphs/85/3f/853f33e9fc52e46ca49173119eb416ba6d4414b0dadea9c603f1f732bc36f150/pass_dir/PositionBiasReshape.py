import torch
import triton
import triton.language as tl


# Pattern matching function for position bias reshape
def pattern(in_3, in_4):
    """
    Position bias pattern: index + view + permute + contiguous + unsqueeze
    in_3: relative_position_bias_table [732, 12 or 16]
    in_4: view indices [38809]
    """
    tmp_2 = in_3[in_4]
    tmp_3 = tmp_2.view(197, 197, -1)
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = tmp_5.unsqueeze(0)
    return tmp_6


# Argument extraction function
def replacement_args(in_3, in_4):
    return (in_3, in_4)


# Optimized Triton kernel for position bias reshape
@triton.jit
def position_bias_kernel(
    table_ptr, indices_ptr, output_ptr,
    n_indices, n_heads, seq_len: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Each program processes one head
    head_idx = tl.program_id(0)
    table_offset = head_idx
    output_offset = head_idx * seq_len * seq_len
    
    # Load all indices for this head
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_indices
    
    # Load indices
    idx = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    # Load from table for this head
    table_offsets = idx * n_heads + head_idx
    values = tl.load(table_ptr + table_offsets, mask=mask, other=0.0)
    
    # Reshape: view to [197, 197]
    # Permute and make contiguous: stored as [seq_len, seq_len]
    # Output will be [1, n_heads, seq_len, seq_len]
    
    # Write to output in the right order for permuted view
    # Original: view to (197, 197, n_heads), permute to (n_heads, 197, 197), unsqueeze to (1, n_heads, 197, 197)
    # We want output [1, n_heads, 197, 197]
    
    # Compute output indices
    out_offsets = output_offset + offsets
    tl.store(output_ptr + out_offsets, values, mask=mask)


@torch.fx.wrap
def triton_position_bias(table, indices):
    """
    Triton implementation for position bias computation
    table: [732, 12] or [732, 16] - relative position bias table
    indices: [38809] - view indices
    """
    n_indices = indices.shape[0]
    n_heads = table.shape[1]
    seq_len = 197
    
    # Calculate expected output shape: [1, n_heads, seq_len, seq_len]
    output_shape = (1, n_heads, seq_len, seq_len)
    output = torch.empty(output_shape, dtype=table.dtype, device=table.device)
    
    # One program per head
    grid = (n_heads,)
    
    BLOCK_SIZE = triton.next_power_of_2(n_indices)
    
    position_bias_kernel[grid](
        table, indices, output,
        n_indices, n_heads, seq_len, BLOCK_SIZE
    )
    
    return output


def replacement_func():
    return triton_position_bias