import torch
import triton
import triton.language as tl

# Pattern matching function for linear transformation
def pattern(in_0, in_3):
    """Pattern: Linear transformation torch.nn.functional.linear(in_3, in_0, None)"""
    linear = torch.nn.functional.linear(in_3, in_0, None)
    return linear

# Argument extraction function
def replacement_args(in_0, in_3):
    return (in_0, in_3)

# Optimized kernel for linear operations
@triton.jit
def optimized_linear_kernel(
    weight_ptr,          # Weight matrix [out_features, in_features]
    input_ptr,          # Input tensor [batch_size, seq_len, in_features]
    output_ptr,         # Output tensor [batch_size, seq_len, out_features]
    batch_size,         # Batch size
    seq_len,            # Sequence length  
    in_features,        # Input features dimension
    out_features,       # Output features dimension
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for seq_len dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for features dimension
):
    # Program identifiers
    pid_m = tl.program_id(0)  # batch dimension  
    pid_n = tl.program_id(1)  # seq_len dimension
    
    # Calculate ranges for this program
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Initialize accumulator for this thread tile
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, out_features), dtype=tl.bfloat16)
    
    # Loop over input features dimension
    for k_offset in range(0, in_features, BLOCK_SIZE_K):
        k_end = min(k_offset + BLOCK_SIZE_K, in_features)
        
        # Load weight block [out_features, k_end-k_offset]
        weight_block = tl.load(
            weight_ptr + (0, k_offset),
            mask=(k_offset < in_features)[None, :],
            other=0.0
        )
        
        # Process all elements in this thread tile batch and sequence range
        for m_local in range(BLOCK_SIZE_M):
            for n_local in range(BLOCK_SIZE_N):
                m_pos = m_start + m_local
                n_pos = n_start + n_local
                
                if m_pos < batch_size and n_pos < seq_len:
                    # Load input block [in_features]
                    input_block = tl.load(
                        input_ptr + (m_pos, n_pos, k_offset),
                        mask=k_offset < in_features,
                        other=0.0
                    )
                    
                    # Matrix multiplication: [in_features] @ [in_features, out_features]
                    # Result: [out_features]
                    weighted = input_block[None, :] * weight_block.t()  # Transpose for efficient computation
                    
                    # Accumulate result
                    accumulator[m_local, n_local, :] += weighted[0, :]
    
    # Store accumulated results
    for m_local in range(BLOCK_SIZE_M):
        for n_local in range(BLOCK_SIZE_N):
            for k_local in range(out_features):
                m_pos = m_start + m_local
                n_pos = n_start + n_local
                k_pos = k_local
                
                if (m_pos < batch_size and 
                    n_pos < seq_len and 
                    k_pos < out_features):
                    tl.store(
                        output_ptr + (m_pos, n_pos, k_pos),
                        accumulator[m_local, n_local, k_pos]
                    )

@torch.fx.wrap
def optimized_linear(weight, input_tensor):
    """Optimized linear transformation using Triton"""
    # Get tensor shapes
    batch_size, seq_len, in_features = input_tensor.shape
    hidden_dim, out_features = weight.shape
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, out_features), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Choose block sizes based on tensor dimensions
    BLOCK_SIZE_M = min(32, batch_size)
    BLOCK_SIZE_N = min(32, seq_len)
    BLOCK_SIZE_K = 32  # Features per thread
    
    # Calculate grid dimensions
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = 1  # No blocking in features dimension for this simple kernel
    
    # Launch kernel
    optimized_linear_kernel[(grid_m, grid_n, 1)](
        weight_ptr=weight,
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return optimized_linear