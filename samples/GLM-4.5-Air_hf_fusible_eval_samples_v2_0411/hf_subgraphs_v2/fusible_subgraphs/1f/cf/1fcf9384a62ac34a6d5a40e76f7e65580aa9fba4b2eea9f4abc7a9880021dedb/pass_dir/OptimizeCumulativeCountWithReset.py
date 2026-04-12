import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    # Match the exact computation pattern from model.py
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel
@triton.jit
def optimized_cumulative_count_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch
    batch_idx = tl.program_id(0)
    col = tl.program_id(1)
    
    # Calculate base offset for this batch
    batch_offset = batch_idx * seq_len
    
    # Load input element
    input_val = tl.load(input_ptr + batch_offset + col, mask=col < seq_len, other=0)
    
    # Create mask: 1 if input != 1, 0 if input == 1
    mask = (input_val != 1).to(tl.int64)
    
    if col == 0:
        # For first position, output is 2 if input != 1, 1 if input == 1
        # Which is: input_val if input_val != 1 else 1
        final_val = tl.where(mask == 1, input_val, 1)
    else:
        # We need to compute cumulative sum along the sequence, then multiply by mask, then add 1
        # Since this is sequential, we need to compute it step by step
        
        # Load previous cumulative sum value
        prev_cumsum = tl.load(output_ptr + batch_offset + col - 1, mask=col - 1 < seq_len, other=0) - 1  # Subtract 1 to get the actual value
        
        if col >= 2:
            # Get the mask value from 2 positions back
            prev_mask = tl.load(input_ptr + batch_offset + col - 2, mask=col - 2 < seq_len, other=0)
            prev_mask_int = (prev_mask != 1).to(tl.int64)
            
            # If previous position was a reset (input == 1), then current position starts fresh
            # Otherwise, increment the previous cumulative value
            output_val = tl.where(prev_mask_int == 1, prev_cumsum + 1, 1)
        else:
            # For position 1, just use the current input if != 1, otherwise 1
            output_val = tl.where(mask == 1, input_val, 1)
        
        final_val = output_val + 1
    
    # Store result
    tl.store(output_ptr + batch_offset + col, final_val, mask=col < seq_len)

# Kernel wrapper
@torch.fx.wrap
def optimized_cumulative_count_wrapper(in_0):
    batch_size, seq_len = in_0.shape
    
    # Use a reasonable block size (but each thread handles one element)
    BLOCK_SIZE = 1
    
    # Calculate grid dimensions
    num_batches = batch_size
    num_seqs = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Initialize output tensor
    output = torch.empty_like(in_0, dtype=torch.int64)
    
    # Launch kernel
    optimized_cumulative_count_kernel[(num_batches, num_seqs)](
        input_ptr=in_0,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_cumulative_count_wrapper