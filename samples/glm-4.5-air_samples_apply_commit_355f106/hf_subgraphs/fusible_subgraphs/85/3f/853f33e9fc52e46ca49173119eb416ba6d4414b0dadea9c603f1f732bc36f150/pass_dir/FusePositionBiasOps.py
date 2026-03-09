import torch
import triton
import triton.language as tl

def pattern(pos_bias_table, indices):
    # Pattern: index + view + permute + contiguous + unsqueeze
    indexed = pos_bias_table[indices]
    reshaped = indexed.view(197, 197, -1)
    permuted = reshaped.permute(2, 0, 1)
    contiguous = permuted.contiguous()
    result = contiguous.unsqueeze(0)
    return result

def replacement_args(pos_bias_table, indices):
    return (pos_bias_table, indices)

@triton.jit
def fused_position_bias_kernel(
    pos_bias_table_ptr,
    indices_ptr,
    out_ptr,
    bias_table_shape0,  # 732
    indices_len,        # 38809
    HEADS: tl.constexpr,  # 12 or 16
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < indices_len
    
    # Load indices
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    # Convert linear indices to 2D coordinates and head mapping
    row = indices // 197
    col = indices % 197
    head_idx = indices // 197
    head_idx = tl.minimum(head_idx, bias_table_shape0 - 1)
    
    # For each head, process indices that belong to it
    for head in range(HEADS):
        # Find indices that belong to this specific head
        head_mask = (head_idx == head) & mask
        
        # Precompute output base offset for this head
        base_output_offset = head * (197 * 197)
        
        # For each feature position, load and store the bias value
        for feat in range(HEADS):
            # Load the bias value for this head and feature
            bias_offset = head * 16 + feat  # Use 16 for stride to avoid power-of-2 issues
            bias_value = tl.load(pos_bias_table_ptr + bias_offset)
            
            # Compute output offset: [head, feature, row, col] in flattened layout
            # Stride order: feature moves slowest within head, then row, then col
            output_offset = base_output_offset + feat * (197 * 197) + row * 197 + col
            
            # Store the bias value using appropriate mask
            # If head_mask is all False, no stores will occur
            tl.store(out_ptr + output_offset, bias_value, mask=head_mask)

@torch.fx.wrap
def fused_position_bias(pos_bias_table, indices):
    # Output shape: [1, heads, 197, 197]
    heads = pos_bias_table.shape[1]
    output_shape = (1, heads, 197, 197)
    out = torch.empty(output_shape, dtype=pos_bias_table.dtype, device=pos_bias_table.device)
    
    # Flatten indices for processing
    indices_flat = indices.flatten()
    
    BLOCK_SIZE = 1024
    num_programs = (indices_flat.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use the unified kernel with the correct number of heads
    fused_position_bias_kernel[(num_programs,)](
        pos_bias_table,
        indices,
        out,
        pos_bias_table.shape[0],
        indices_flat.numel(),
        heads,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_position_bias