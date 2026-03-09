import torch
import triton
import triton.language as tl

# Pattern matching function for mean reduction along dimension -2
def pattern(input_tensor):
    # Match input_tensor.mean(-2) - mean reduction along the sequence dimension
    output = input_tensor.mean(-2)
    return output

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized Triton kernel for mean reduction along dimension -2
@triton.jit
def mean_kernel(
    input_ptr, 
    output_ptr,
    batch_size,
    seq_len,  # This is the dimension being reduced (dim=-2)
    features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program processes a block of the output matrix
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges this program will handle
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min(m_start + BLOCK_SIZE_M, batch_size)
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min(n_start + BLOCK_SIZE_N, features)
    
    # Initialize accumulator for this block
    accumulator = tl.zeros((m_end - m_start, n_end - n_start), dtype=tl.float32)
    
    # Each thread handles one element in the current block
    for m in range(m_start, m_end):
        for n in range(n_start, n_end):
            # Sum over the sequence dimension (dim=-2)
            total = 0.0
            for s in range(seq_len):
                # Calculate input: [batch_size, seq_len, features]
                offset = (m * seq_len + s) * features + n
                total += tl.load(input_ptr + offset)
            
            # Calculate mean by dividing by sequence length
            mean_val = total / seq_len
            output_offset = ((m - m_start) * features + (n - n_start))
            tl.store(output_ptr + output_offset, mean_val)

# Optimized Triton kernel for mean reduction along dimension -2
@triton.jit
def mean_kernel(
    input_ptr, 
    output_ptr,
    batch_size,
    seq_len,
    features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one row of the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute bounds
    m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize sum accumulator
    sum_val = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute valid bounds
    m_mask = m < batch_size
    n_mask = n < features
    
    # Sum over sequence dimension (dim=-2)
    for s in range(seq_len):
        # Load input tensor [batch_size, seq_len, features]
        # Index calculation: (batch_idx * seq_len + seq_idx) * features + feat_idx
        input_vals = tl.load(input_ptr + (m[:, None] * seq_len + s) * features + n[None, :], 
                           mask=m_mask[:, None] and n_mask[None, :], 
                           other=0.0)
        sum_val += input_vals
    
    # Calculate mean
    mean_val = sum_val / seq_len
    
    # Store result
    tl.store(output_ptr + m[:, None] * features + n[None, :], 
             mean_val, 
             mask=m_mask[:, None] and n_mask[None, :])

# Kernel wrapper
@torch.fx.wrap
def optimized_mean(input_tensor):
    batch_size, seq_len, features = input_tensor.shape
    
    # Determine block sizes for optimal GPU utilization
    BLOCK_SIZE_M = 32  # Block size for batch dimension
    BLOCK_SIZE_N = 64  # Block size for features dimension
    
    # Calculate grid dimensions
    num_blocks_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_blocks_m, num_blocks_n)
    
    # Create output tensor
    output = torch.empty((batch_size, features), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Launch optimized kernel
    mean_kernel[grid](
        input_tensor,
        output,
        batch_size,
        seq_len,
        features,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_mean