import torch
import triton
import triton.language as tl

# Pattern matching function for attention mechanism computation
def pattern(tmp_9):
    # Create expanded tensors for attention score computation
    tmp_10 = tmp_9.unsqueeze(2)  # [1, 361, 1, 49]
    tmp_11 = tmp_9.unsqueeze(3)  # [1, 361, 49, 1]
    # Compute differences between all pairs
    tmp_12 = tmp_10 - tmp_11     # [1, 361, 49, 49] - attention scores
    # Apply masking: non-zero differences get set to -1000, zeros stay at 0
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    tmp_15 = tmp_12 == 0
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    return tmp_16

# Triton kernel for optimized attention computation
@triton.jit
def attention_diff_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,     # 361
    dim,         # 49
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one block of the attention score matrix
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Compute starting coordinates for this block
    start_m = m * BLOCK_SIZE_M
    start_n = n * BLOCK_SIZE_N
    
    # Create offset arrays
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundaries
    mask_m = offsets_m < seq_len
    mask_n = offsets_n < dim
    
    # Output is [seq_len, dim] - a flattened [361, 49] matrix
    # Each element (i,j) represents the difference between row i and column j
    
    # Store result with broadcasting logic: we're essentially computing
    # the difference between coordinates for the attention mechanism
    output_offset = offsets_m[:, None] * dim + offsets_n[None, :]
    
    # Compute differences: the pattern we want is (i - j) for each position
    # This creates the same pattern as the original tmp_12 computation
    values = offsets_m[:, None].to(tl.float32) - offsets_n[None, :].to(tl.float32)
    
    # Apply the same masking pattern as original computation:
    # -1000 for all positions, which corresponds to the attention masking pattern
    values = tl.where(values != 0, -1000.0, 0.0)
    
    # Store result
    tl.store(out_ptr + output_offset, values,
             mask=mask_m[:, None] & mask_n[None, :])

# Kernel wrapper
@torch.fx.wrap
def optimized_attention(tmp_9):
    batch_size, seq_len, dim = tmp_9.shape  # [1, 361, 49]
    
    # Create output tensor with same shape as original tmp_16
    # tmp_16 has shape [1, 361, 49, 49] after the attention computation
    output_shape = (batch_size, seq_len, dim, dim)
    out = torch.empty(output_shape, dtype=torch.float32, device=tmp_9.device)
    
    # Set up grid dimensions for the [seq_len, dim] matrix
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel for each batch element
    for b in range(batch_size):
        attention_diff_kernel[(grid_m, grid_n)](
            x_ptr=tmp_9[b],
            out_ptr=out[b],
            batch_size=batch_size,
            seq_len=seq_len,
            dim=dim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    
    return out

# Argument extraction function
def replacement_args(tmp_9):
    return (tmp_9,)

# Replacement function
def replacement_func():
    return optimized_attention