import torch
import triton
import triton.language as tl

# Pattern matching function for element-wise multiplication with broadcasting
def pattern(in_2, in_1):
    """Pattern: Element-wise multiplication with broadcasting in_2 * in_1"""
    tmp_3 = in_2 * in_1
    return tmp_3

# Argument extraction function
def replacement_args(in_2, in_1):
    return (in_2, in_1)

# Optimized kernel for multiplication with broadcasting
@triton.jit
def optimized_multiply_broadcast_kernel(
    tensor_ptr,          # Input tensor [batch_size, seq_len, features]
    scale_ptr,           # Scale tensor [features] (or scalar if ndim == 0)
    output_ptr,          # Output tensor [batch_size, seq_len, features]
    batch_size,          # Batch size
    seq_len,             # Sequence length
    features,            # Features dimension
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for seq_len dimension  
    BLOCK_SIZE_K: tl.constexpr,  # Block size for features dimension
):
    # Program identifiers
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # seq_len dimension
    pid_k = tl.program_id(2)  # features dimension
    
    # Calculate ranges for this program
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    k_start = pid_k * BLOCK_SIZE_K
    k_end = min(k_start + BLOCK_SIZE_K, features)
    
    # Process all elements in this thread tile
    for k_local in range(k_start, k_end):
        # Load scale value
        if scale_ptr.ndim == 0:  # Scalar case
            scale_value = tl.load(scale_ptr, other=0.0)
        elif scale_ptr.shape[0] == features:  # [features] case
            scale_value = tl.load(scale_ptr + k_local, other=0.0)
        else:  # Shouldn't happen for our pattern, but handle gracefully
            scale_value = tl.load(scale_ptr + k_local % scale_ptr.shape[0], other=0.0)
        
        # Process all batch and sequence positions for this feature
        for m_local in range(BLOCK_SIZE_M):
            for n_local in range(BLOCK_SIZE_N):
                m_pos = m_start + m_local
                n_pos = n_start + n_local
                
                if m_pos < batch_size and n_pos < seq_len:
                    # Load tensor element
                    tensor_value = tl.load(
                        tensor_ptr + (m_pos, n_pos, k_local),
                        other=0.0
                    )
                    
                    # Apply scaling
                    result = tensor_value * scale_value
                    
                    # Store result
                    tl.store(
                        output_ptr + (m_pos, n_pos, k_local),
                        result
                    )

@torch.fx.wrap 
def optimized_multiply_broadcast(tensor, scale):
    """Optimized element-wise multiplication with broadcasting"""
    # Get tensor shapes
    batch_size, seq_len, features = tensor.shape
    
    # Create output tensor
    output = torch.empty_like(tensor)
    
    # Choose optimal block sizes
    BLOCK_SIZE_M = min(32, batch_size)
    BLOCK_SIZE_N = min(32, seq_len)
    BLOCK_SIZE_K = 128  # Features per thread
    
    # Calculate grid dimensions
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (features + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Launch kernel
    optimized_multiply_broadcast_kernel[(grid_m, grid_n, grid_k)](
        tensor_ptr=tensor,
        scale_ptr=scale,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        features=features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return optimized_multiply_broadcast