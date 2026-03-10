import torch
import triton
import triton.language as tl

def pattern(expanded_tensor):
    # First max operation over dimension 0 (the expanded dimension with 3 elements)
    max1_result = expanded_tensor.max(0, keepdim=False)
    max1_values = max1_result[0]
    
    # Second max operation over last dimension (with keepdim)
    max2_result = max1_values.max(-1, keepdim=True)
    max2_values = max2_result[0]
    
    # Simple arithmetic operations
    result_plus_1 = max2_values + 1
    final_result = result_plus_1 - 9
    
    return final_result

def replacement_args(expanded_tensor):
    return (expanded_tensor,)

@triton.jit
def optimized_reduction_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output
    pid = tl.program_id(0)
    
    # Calculate range this program handles
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    
    # Load batch and sequence dimensions
    batch_offset = offsets * 1  # No sequence dimension in final result
    
    if pid < batch_size:
        # For each batch position, we need to process sequence length
        # But the final output is just batch size (max over sequence)
        
        # Load the 3 expanded values for this batch position
        # Shape: (3, batch_size, seq_len) -> for each batch, we need max over seq and then max over 3
        max_over_seq = tl.int64(-1)  # Start with minimum value
        
        # For each position in the expanded dimension (3), find max over sequence
        for k in range(3):
            if k < 3:  # Ensure we don't go out of bounds
                offset = (k * batch_size + pid) * seq_len
                seq_vals = tl.load(in_ptr + offset, mask=tl.arange(0, seq_len) < seq_len, other=tl.int64(-1))
                seq_max = tl.max(seq_vals)
                if seq_max > max_over_seq:
                    max_over_seq = seq_max
        
        # Apply arithmetic operations (+1, -9)
        final_val = max_over_seq + 1 - 9
        
        # Store result
        tl.store(out_ptr + pid, final_val)
    else:
        # Fill with zeros for invalid positions
        tl.store(out_ptr + pid, 0, mask=False)

@torch.fx.wrap
def optimized_reduction(expanded_tensor):
    # expanded_tensor has shape (3, batch_size, seq_len)
    num_expanded, batch_size, seq_len = expanded_tensor.shape
    
    # Final result is just batch size
    batch_size = batch_size
    out = torch.empty(batch_size, dtype=torch.int64, device=expanded_tensor.device)
    
    # Determine block size
    BLOCK_SIZE = 256  # Can be tuned for batch size
    num_blocks = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_reduction_kernel[(num_blocks,)](
        in_ptr=expanded_tensor,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_reduction