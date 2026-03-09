import torch
import triton
import triton.language as tl

def pattern(table, indices):
    """Pattern for bias table indexing + view + permute + contiguous + unsqueeze fusion"""
    indexed = table[indices]                                # Indexing: [732, C] -> [38809, C]  
    reshaped = indexed.view(197, 197, -1)                   # Reshape: [38809, C] -> [197, 197, C]
    permuted = reshaped.permute(2, 0, 1)                    # Transpose: [197, 197, C] -> [C, 197, 197]
    contiguous = permuted.contiguous()                     # Make contiguous
    result = contiguous.unsqueeze(0)                        # Add batch: [C, 197, 197] -> [1, C, 197, 197]
    return result

def replacement_args(table, indices):
    return (table, indices)

@triton.jit
def bias_table_kernel(
    table_ptr,
    indices_ptr,
    output_ptr,
    table_rows,
    table_cols,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS: tl.constexpr,
    MAX_CHANNELS: tl.constexpr,
):
    """Optimized kernel for bias table processing fusion"""
    pid = tl.program_id(0)
    output_offset = pid * BLOCK_SIZE
    
    # Handle boundaries
    if output_offset >= n_elements:
        return
    
    # Simple block-wise processing
    mask = output_offset + tl.arange(0, BLOCK_SIZE) < n_elements
    
    # Load indices for this block
    indices_block = tl.load(indices_ptr + output_offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=-1)
    
    # Vectorized table lookup: Load from table and store to output
    idx_vals = indices_block
    
    # Only process valid indices within bounds
    valid_mask = mask & (idx_vals >= 0) & (idx_vals < table_rows)
    
    # Calculate table offsets for valid indices 
    table_offsets = idx_vals * CHANNELS
    
    # Load values from table (use max power-of-2 range and mask)
    channel_mask = tl.arange(0, MAX_CHANNELS) < CHANNELS
    values = tl.load(table_ptr + table_offsets + tl.arange(0, MAX_CHANNELS)[None, :], 
                     mask=valid_mask[:, None] & channel_mask[None, :])
    
    # Store to output (linear layout then reshape will be done in Python)
    output_offsets = output_offset + tl.arange(0, BLOCK_SIZE)[:, None]
    tl.store(output_ptr + output_offsets * MAX_CHANNELS + tl.arange(0, MAX_CHANNELS)[None, :], 
             values, mask=valid_mask[:, None] & channel_mask[None, :])

@torch.fx.wrap
def fused_bias_table_processing(table, indices):
    """Fused kernel wrapper for bias table processing"""
    n_elements = indices.numel()  # 38809
    batch_size = 1
    rows = 197 
    channels = table.shape[1]  # C = 12 or 16
    max_channels = 16  # Must match kernel
    
    # Calculate output shape [38809, max_channels] initially (to match kernel)
    output_shape = (n_elements, max_channels)
    output = torch.zeros(output_shape, dtype=table.dtype, device=table.device)
    
    # Launch kernel - one block for each 1024 elements
    total_blocks = (n_elements + 1023) // 1024
    
    if total_blocks > 0:
        bias_table_kernel[(
            total_blocks,
        )](
            table_ptr=table,
            indices_ptr=indices,
            output_ptr=output,  # Direct linear output
            table_rows=table.shape[0],
            table_cols=channels,
            n_elements=indices.numel(),
            BLOCK_SIZE=1024,
            CHANNELS=channels,
            MAX_CHANNELS=max_channels,
        )
    
    # Slice to actual number of channels and do final reshaping
    # Slice to actual channels: [38809, max_channels] -> [38809, channels]
    sliced_output = output[:, :channels]
    
    # Do the final reshaping operations that were in the original sequence
    # First reshape: [38809, channels] -> [197, 197, channels]
    reshaped = sliced_output.view(rows, rows, channels)
    
    # Then permute: [197, 197, channels] -> [channels, 197, 197]  
    permuted = reshaped.permute(2, 0, 1)
    
    # Add batch dimension: [channels, 197, 197] -> [1, channels, 197, 197]
    final_result = permuted.unsqueeze(0)
    
    return final_result

def replacement_func():
    return fused_bias_table_processing