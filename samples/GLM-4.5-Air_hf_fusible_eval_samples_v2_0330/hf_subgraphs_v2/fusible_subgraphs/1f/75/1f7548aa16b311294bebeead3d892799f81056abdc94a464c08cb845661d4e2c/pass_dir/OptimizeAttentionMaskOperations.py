import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Attention mask processing with range operations - match exactly from original
    tmp_2 = in_0.to(dtype=torch.bool)
    # We'll handle the arange in the optimized kernel instead of pattern
    # For now, just match the basic operations
    
    tmp_3 = torch.arange(3)  # This is just for pattern - actual handled in optimization
    tmp_3 += 0
    tmp_4 = tmp_3
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    
    # Range and mask creation
    tmp_7 = torch.arange(3)  # This is just for pattern - actual handled in optimization
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = tmp_7 <= tmp_8
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_11 * tmp_12
    
    return tmp_13

def replacement_args(in_0, in_1, in_2, in_3):
    # Return all original arguments plus the sequence length
    seq_len = in_0.shape[1]
    batch_size = in_0.shape[0]
    return (in_0, in_1, in_2, in_3, seq_len, batch_size)

@triton.jit
def optimized_attention_mask_kernel(
    attention_mask_ptr,
    cache_position_ptr,
    output_ptr,
    seq_len: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized kernel that processes attention mask and cache position operations efficiently.
    
    Original operations:
    1. Convert attention_mask to boolean
    2. Create arange(seq_len) 
    3. Index boolean mask with arange
    4. Create another arange(seq_len)
    5. Compare arange with cache_position.view(-1, 1)
    6. Add dimensions and expand
    7. Multiply with indexed mask
    
    Optimized kernel:
    - Processes everything in single efficient pass
    - Avoids multiple kernel launches for arange and indexing
    - Optimizes memory access patterns
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE_M
    
    # Create range for this block
    range_local = tl.arange(0, BLOCK_SIZE_M)
    pos = offset + range_local
    mask = pos < seq_len
    
    # Process attention mask: convert to boolean and index with range
    if attention_mask_ptr is not None:
        # Get attention mask slice: [batch, seq_len] -> [batch, seq_len] 
        attention_mask_base = tl.load(attention_mask_ptr + tl.arange(0, batch_size)[:, None] * seq_len + pos[None, :], 
                                    mask=mask[None, :], other=False)
        attention_mask_bool = attention_mask_base.to(tl.int1)  # Convert to boolean
    else:
        attention_mask_bool = True
    
    # Process cache position comparison
    if cache_position_ptr is not None:
        # Load cache position and add dimensions for broadcasting
        cache_pos = tl.load(cache_position_ptr + pos, mask=mask, other=0)
        cache_pos_2d = cache_pos[:, None]  # Shape: [BLOCK_SIZE_M, 1]
        
        # Create comparison: range <= cache_position
        range_2d = pos[:, None]  # Shape: [BLOCK_SIZE_M, 1]
        comparison_result = range_2d <= cache_pos_2d
        comparison_result = comparison_result.to(tl.int32)  # Convert to int32 for multiplication
    else:
        comparison_result = 1
    
    # Multiply: attention_mask * comparison_result
    # This gives us the final masked and comparison result
    final_result = attention_mask_bool * comparison_result
    
    # Store result with proper broadcasting dimensions
    # Add extra dimensions: [batch, seq_len] -> [batch, 1, 1, seq_len]
    for b in range(batch_size):
        output_offset = b * BLOCK_SIZE_M + offset
        tl.store(output_ptr + output_offset, final_result, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3, seq_len, batch_size):
    """
    Wrapper function that launches the optimized attention mask kernel.
    """
    # Create output tensor
    output_shape = (batch_size, 1, 1, seq_len)
    dtype = torch.int32  # We'll use int32 for multiplication results
    output = torch.empty(output_shape, dtype=dtype)
    
    # Set up Triton kernel configuration
    BLOCK_SIZE = 256  # Optimal block size for GPU
    
    # Calculate number of programs needed
    num_programs = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_attention_mask_kernel[(num_programs,)](
        in_0,  # attention_mask
        in_2,  # cache_position
        output,
        seq_len,
        batch_size,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return kernel_wrapper