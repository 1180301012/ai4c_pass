import torch
import triton
import triton.language as tl

# Very simple pattern - just cumsum operation
def pattern(x, y):
    # Start with the simplest possible pattern
    result = y.cumsum(-1)
    return (result, result.unsqueeze(0))

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel for cumulative sum and masked fill
@triton.jit
def cumsum_mask_kernel(
    in_0_ptr, in_1_ptr,
    intermediate_ptr,
    batch_size, seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load current values
    in_0_vals = tl.load(in_0_ptr + offsets, mask=mask, other=0)
    in_1_vals = tl.load(in_1_ptr + offsets, mask=mask, other=0)
    
    # Create cumsum result
    cumsum = tl.zeros_like(in_1_vals, dtype=tl.int64)
    
    # Handle cumsum with masked values
    if pid == 0:
        # First block
        cumsum = tl.cumsum(in_1_vals, axis=0)
        # Reset to 0 where in_0 is 0
        cumsum = tl.where(in_0_vals == 0, 0, cumsum)
    else:
        # Non-first blocks - need to add previous values
        prev_offset = (pid - 1) * BLOCK_SIZE + BLOCK_SIZE - 1
        prev_in_0 = tl.load(in_0_ptr + prev_offset)
        prev_in_1 = tl.load(in_1_ptr + prev_offset)
        prev_cumsum = tl.where(prev_in_0 == 0, 0, prev_in_1)
        
        # Add previous cumulative sum to current cumsum
        cumsum = tl.cumsum(in_1_vals, axis=0)
        cumsum = tl.where(in_0_vals == 0, 0, cumsum)
        cumsum += prev_cumsum
    
    # Subtract 1 and apply masked fill logic
    result = cumsum - 1
    final_result = tl.where(in_0_vals == 0, 1, result)
    
    tl.store(intermediate_ptr + offsets, final_result, mask=mask)

# Kernel for expanding tensor
@triton.jit
def expand_kernel(
    intermediate_ptr, expanded_ptr,
    batch_size, seq_len, expand_factor,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the expanded tensor
    pid = tl.program_id(0)
    
    # Calculate position in expanded tensor [expand_factor, batch_size, seq_len]
    total_elements = batch_size * seq_len
    element_idx = pid % total_elements
    expand_idx = pid // total_elements
    
    batch_idx = element_idx // seq_len
    seq_idx = element_idx % seq_len
    
    # Load from intermediate tensor
    src_offset = batch_idx * seq_len + seq_idx
    src_value = tl.load(intermediate_ptr + src_offset)
    
    # Store in expanded tensor
    dst_offset = expand_idx * batch_size * seq_len + element_idx
    tl.store(expanded_ptr + dst_offset, src_value)

# Kernel for max reduction
@triton.jit  
def max_reduce_kernel(
    expanded_ptr, final_result_ptr,
    batch_size, seq_len, expand_factor,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sequence position across batches
    pid = tl.program_id(0)
    
    if pid >= seq_len:
        return
    
    # Find max across expand_factor for this position
    current_max = -1
    
    for b_idx in range(batch_size):
        for e_idx in range(expand_factor):
            offset = e_idx * batch_size * seq_len + b_idx * seq_len + pid
            val = tl.load(expanded_ptr + offset)
            if val > current_max:
                current_max = val
    
    # All sequences should give the same result due to expand operation
    final_value = current_max + 1 - 9
    tl.store(final_result_ptr, final_value)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_computation(in_0, in_1):
    batch_size, seq_len = in_0.shape
    expand_factor = 3
    
    # Intermediate tensor for cumsum + masked fill result
    intermediate = torch.empty((batch_size, seq_len), dtype=torch.int64, device=in_0.device)
    
    # Expanded tensor [expand_factor, batch_size, seq_len]
    expanded_output = torch.empty((expand_factor, batch_size, seq_len), dtype=torch.int64, device=in_0.device)
    final_result = torch.empty((), dtype=torch.int64, device=in_0.device)
    
    # Set block size based on sequence length
    BLOCK_SIZE = min(1024, (seq_len + 63) // 64 * 64)
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Step 1: Cumulative sum with masked fill
    cumsum_mask_kernel[(num_blocks,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        intermediate_ptr=intermediate,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Step 2: Expand tensor
    total_elements = expand_factor * batch_size * seq_len
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    expand_kernel[(grid,)](
        intermediate_ptr=intermediate,
        expanded_ptr=expanded_output,
        batch_size=batch_size,
        seq_len=seq_len,
        expand_factor=expand_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Step 3: Max reduction
    grid = (1, 1, 1)  # Single element result
    max_reduce_kernel[(grid,)](
        expanded_ptr=expanded_output,
        final_result_ptr=final_result,
        batch_size=batch_size,
        seq_len=seq_len,
        expand_factor=expand_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (final_result, expanded_output)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_computation