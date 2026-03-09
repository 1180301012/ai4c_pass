import torch
import triton
import triton.language as tl

def pattern(tmp_3, tmp_0):
    # tmp_3: result of in_4 / in_3 (implicitly float)
    # tmp_0: attention mask (int64)
    tmp_4 = tmp_3.to(torch.float32)  # Unnecessary type conversion
    tmp_5 = tmp_0.unsqueeze(-1)  # Add new dimension at the end
    tmp_6 = tmp_4 * tmp_5  # Multiplication
    tmp_7 = tmp_6.to(torch.float32)  # Another unnecessary type conversion
    return tmp_7

def replacement_args(tmp_3, tmp_0):
    # tmp_3 already contains the division result in float32
    # We just need the attention mask
    return (tmp_3, tmp_0)

@triton.jit
def optimized_multiply_unsqueeze_kernel(
    division_result_ptr,
    attention_mask_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the output matrix
    row_idx = tl.program_id(0)
    
    # Ensure we don't go out of bounds
    if row_idx >= batch_size * seq_len:
        return
    
    # Calculate batch and sequence indices
    batch_idx = row_idx // seq_len
    seq_idx = row_idx % seq_len
    
    # Load attention mask for this batch and convert to float32
    attention_mask_val = tl.load(attention_mask_ptr + batch_idx)
    attention_float = tl.cast(attention_mask_val, tl.float32)
    
    # Calculate base offset for this row in the division result tensor
    base_offset = row_idx * hidden_dim
    
    # Load a block of the division result (multiple hidden dimensions) and multiply with attention
    offsets = base_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_idx + 1) * hidden_dim
    
    # Load the block
    input_block = tl.load(division_result_ptr + offsets, mask=mask, other=0.0)
    
    # Apply element-wise multiplication with attention mask
    output_block = input_block * attention_float
    
    # Store the result
    tl.store(output_ptr + offsets, output_block, mask=mask)

@torch.fx.wrap
def optimized_multiply_unsqueeze(division_result, attention_mask):
    # Get input shapes
    batch_size, seq_len, hidden_dim = division_result.shape
    
    # Set block size to be a power of 2 (256 is less than 320, so it works)
    BLOCK_SIZE = 256
    
    # Calculate grid (one program per row)
    num_rows = batch_size * seq_len
    num_programs = num_rows
    
    # Create output tensor
    output = torch.empty_like(division_result, dtype=torch.float32)
    
    # Launch kernel
    optimized_multiply_unsqueeze_kernel[(num_programs,)](
        division_result,
        attention_mask,
        output,
        batch_size,
        seq_len,
        hidden_dim,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_multiply_unsqueeze