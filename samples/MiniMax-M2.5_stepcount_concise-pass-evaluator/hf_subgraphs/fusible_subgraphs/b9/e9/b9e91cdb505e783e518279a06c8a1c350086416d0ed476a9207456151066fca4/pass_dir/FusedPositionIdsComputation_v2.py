import torch
import triton
import triton.language as tl


# Pattern matching function - matches the position IDs computation pattern
def pattern(in_0):
    """
    Match the computation pattern for position IDs:
    1. Create mask for non-padding tokens (token != 1)
    2. Convert to int
    3. Compute cumsum along sequence dimension
    4. Multiply by mask and add 1
    """
    tmp_1 = in_0.ne(1)  # Create mask: tokens != 1 (padding)
    tmp_2 = tmp_1.int()  # Convert boolean to int
    tmp_3 = torch.cumsum(tmp_2, dim=1)  # Cumulative sum along sequence
    tmp_4 = tmp_3.type_as(tmp_2)  # Type conversion (no-op, both int)
    tmp_5 = tmp_4 + 0  # No-op addition
    tmp_6 = tmp_5 * tmp_2  # Multiply cumsum by mask
    tmp_7 = tmp_6.long()  # Convert to long
    tmp_8 = tmp_7 + 1  # Add 1 for 1-indexed positions
    return tmp_8


def replacement_args(in_0):
    return (in_0,)


# Optimized Triton kernel: one thread block per row for efficient cumsum
# With autotuning for better performance

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=8),
    ],
    key=['seq_len'],
)
@triton.jit
def position_ids_per_row_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block handles one row (one batch element)
    batch_idx = tl.program_id(0)
    
    if batch_idx >= batch_size:
        return
    
    row_offset = batch_idx * seq_len
    
    # Compute cumsum for this row
    cumsum = 0
    for j in range(seq_len):
        offset = row_offset + j
        token = tl.load(input_ptr + offset)
        mask = (token != 1).to(tl.int32)
        cumsum = cumsum + mask
        
        # Compute final result: (cumsum * mask).long() + 1
        result = (cumsum * mask).to(tl.int64) + 1
        tl.store(output_ptr + offset, result)


@torch.fx.wrap
def optimized_position_ids(input_tensor):
    """
    Optimized position IDs computation.
    Uses one thread block per row to efficiently compute cumsum.
    """
    batch_size, seq_len = input_tensor.shape
    
    # Allocate output tensor
    output = torch.empty_like(input_tensor, dtype=torch.long)
    
    # Launch kernel: one thread block per row
    # BLOCK_SIZE is handled by autotune
    position_ids_per_row_kernel[(batch_size,)](
        input_tensor, output, batch_size, seq_len
    )
    
    return output


def replacement_func():
    return optimized_position_ids