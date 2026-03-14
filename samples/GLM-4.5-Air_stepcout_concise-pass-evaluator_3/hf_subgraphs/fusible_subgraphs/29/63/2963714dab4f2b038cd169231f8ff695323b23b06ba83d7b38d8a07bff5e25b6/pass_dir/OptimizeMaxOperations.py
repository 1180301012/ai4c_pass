import torch
import triton
import triton.language as tl

def pattern(expanded_tensor):
    # Pattern matching: sequential max operations with arithmetic
    tmp_8 = expanded_tensor.max(0, keepdim=False)
    tmp_9 = tmp_8[0]
    tmp_10 = tmp_9.max(-1, keepdim=True)
    tmp_11 = tmp_10[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return tmp_13

@triton.jit
def optimized_max_reduce_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch for the final reduction
    batch_idx = tl.program_id(0)
    
    # Initialize max values across expansion dimension (dim 0) and sequence dimension (dim -1)
    max_across_expand = -float('inf')
    max_across_seq = -float('inf')
    
    # Find max across all expansion dimensions (3 copies) and sequence dimension
    for expand_idx in range(3):
        for seq_idx in range(seq_len):
            offset = expand_idx * batch_size * seq_len + batch_idx * seq_len + seq_idx
            val = tl.load(input_ptr + offset, dtype=tl.float32)
            
            # Update max across expansion (dim 0)
            if val > max_across_expand:
                max_across_expand = val
            
            # Update max across sequence (dim -1) 
            if val > max_across_seq:
                max_across_seq = val
    
    # Apply arithmetic: (max_across_seq + 1) - 9 = max_across_seq - 8
    final_result = max_across_seq - 8
    
    # Store final result
    output_offset = batch_idx
    tl.store(output_ptr + output_offset, final_result)

@torch.fx.wrap
def optimized_max_operations(expanded_tensor):
    batch_size, seq_len = expanded_tensor.shape[1], expanded_tensor.shape[2]
    
    # Create output tensor for final results
    output = torch.empty((batch_size,), dtype=expanded_tensor.dtype, device=expanded_tensor.device)
    
    # Calculate grid dimensions
    grid = (batch_size,)
    
    # Launch optimized kernel that performs both max reductions and arithmetic in one pass
    optimized_max_reduce_kernel[grid](
        input_ptr=expanded_tensor,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=1,
    )
    
    return output

def replacement_args(expanded_tensor):
    return (expanded_tensor,)

def replacement_func():
    return optimized_max_operations