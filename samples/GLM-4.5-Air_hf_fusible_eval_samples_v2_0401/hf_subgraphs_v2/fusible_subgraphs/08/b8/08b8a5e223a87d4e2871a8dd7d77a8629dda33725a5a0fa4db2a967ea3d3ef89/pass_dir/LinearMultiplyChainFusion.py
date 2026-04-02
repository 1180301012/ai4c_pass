import torch
import triton
import triton.language as tl

# Pattern matching function for linear + multiply chain
def pattern(in_0, in_1, in_2):
    """Pattern: Linear(in_2, in_0, None) followed by in_1 * linear_result"""
    linear = torch.nn.functional.linear(in_2, in_0, None)
    tmp_2 = in_1 * linear
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized kernel for fused linear + multiply operations
@triton.jit
def linear_multiply_fused_kernel(
    weight_ptr,           # Weight matrix [hidden_dim, in_features]
    multiplier_ptr,       # Multiplier tensor [out_features] or [batch, seq_len, out_features] 
    input_ptr,           # Input tensor [batch, seq_len, in_features]
    output_ptr,          # Output tensor [batch, seq_len, out_features]
    batch_size,          # Batch size
    seq_len,             # Sequence length
    in_features,         # Input features dimension
    out_features,        # Output features dimension
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for seq_len dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for features dimension
    BLOCK_SIZE_FEATURES: tl.constexpr,  # Block size for features in inner loop
):
    # Program identifiers
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # seq_len dimension
    pid_k = tl.program_id(2)  # output feature dimension (blocked)
    
    # Calculate ranges for this program
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    k_start = pid_k * BLOCK_SIZE_FEATURES
    k_end = min(k_start + BLOCK_SIZE_FEATURES, out_features)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_FEATURES), dtype=tl.bfloat16)
    
    # Loop over input features dimension
    for k_offset in range(0, in_features, BLOCK_SIZE_K):
        k_end_local = min(k_offset + BLOCK_SIZE_K, in_features)
        
        # Load weight block [k_end_local-k_offset, k_end_local-k_start]
        weight_block = tl.load(
            weight_ptr + (k_offset, k_start),
            mask=(k_offset < in_features)[:, None] & (k_start < k_end)[None, :],
            other=0.0
        )
        
        # Load multiplier block (with broadcasting)
        if multiplier_ptr.shape[0] == out_features:  # [out_features] case
            multiplier_block = tl.load(
                multiplier_ptr + k_start,
                mask=k_start < k_end,
                other=0.0
            ).to(tl.bfloat16)
        else:  # [batch, seq_len, out_features] case
            multiplier_block = tl.load(
                multiplier_ptr + (m_start, n_start, k_start),
                mask=(m_start < batch_size) & (n_start < seq_len) & (k_start < k_end),
                other=0.0
            )
        
        # Load input blocks for the entire batch and sequence range
        for m_local in range(BLOCK_SIZE_M):
            for n_local in range(BLOCK_SIZE_N):
                m_pos = m_start + m_local
                n_pos = n_start + n_local
                
                if m_pos < batch_size and n_pos < seq_len:
                    # Load input [in_features]
                    input_block = tl.load(
                        input_ptr + (m_pos, n_pos, k_offset),
                        mask=k_offset < in_features,
                        other=0.0
                    )
                    
                    # Multiply and accumulate
                    # weight_block shape: [BLOCK_SIZE_K, BLOCK_SIZE_FEATURES]
                    # input_block shape: [BLOCK_SIZE_K]  
                    # result shape: [BLOCK_SIZE_FEATURES]
                    weighted = input_block[:, None] * weight_block
                    
                    # Add to accumulator and apply multiplier
                    if multiplier_ptr.shape[0] == out_features:  # [out_features] -> broadcast to full tensor
                        multiplier_tile = multiplier_block[None, :]  # [1, BLOCK_SIZE_FEATURES]
                        accumulator[m_local, n_local, :] += weighted * multiplier_tile
                    else:  # [batch, seq_len, out_features] -> specific element
                        multiplier_element = multiplier_block.to(tl.bfloat16)
                        accumulator[m_local, n_local, :] += weighted * multiplier_element
    
    # Store result
    for m_local in range(BLOCK_SIZE_M):
        for n_local in range(BLOCK_SIZE_N):
            for k_local in range(BLOCK_SIZE_FEATURES):
                m_pos = m_start + m_local
                n_pos = n_start + n_local
                k_pos = k_start + k_local
                
                if (m_pos < batch_size and 
                    n_pos < seq_len and 
                    k_pos < k_end):
                    tl.store(
                        output_ptr + (m_pos, n_pos, k_pos),
                        accumulator[m_local, n_local, k_local]
                    )

@torch.fx.wrap
def fused_linear_multiply(weight, multiplier, input_tensor):
    """Fused linear transformation and element-wise multiplication"""
    # Get tensor shapes
    batch_size, seq_len, in_features = input_tensor.shape
    hidden_dim, out_features = weight.shape
    
    # Ensure output shape matches multiplier's broadcasting needs
    if multiplier.ndim == 1:
        # Multiplier is [out_features], result will be [batch, seq_len, out_features]
        output_shape = (batch_size, seq_len, out_features)
    else:
        # Multiplier already matches output shape
        output_shape = multiplier.shape
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Choose block sizes based on tensor dimensions
    BLOCK_SIZE_M = min(32, batch_size)
    BLOCK_SIZE_N = min(32, seq_len) 
    BLOCK_SIZE_FEATURES = 128  # Features per thread
    
    # Calculate grid dimensions
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (out_features + BLOCK_SIZE_FEATURES - 1) // BLOCK_SIZE_FEATURES
    
    # Launch kernel
    linear_multiply_fused_kernel[(grid_m, grid_n, grid_k)](
        weight_ptr=weight,
        multiplier_ptr=multiplier,
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=32,
        BLOCK_SIZE_FEATURES=BLOCK_SIZE_FEATURES,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return fused_linear_multiply