import torch
import triton
import triton.language as tl

# Pattern matching function - must match exactly what's in model.py
def pattern(in_0, in_1, in_2):
    # Linear transformation: in_2 @ in_1.T + in_0
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    
    # Reshape from [1, 19, 18] to [19, 9, 1]
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    
    # Softmax along dimension 1 (size 9)
    tmp_4 = torch.softmax(tmp_3, dim=1)
    
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized Triton kernel
@triton.jit
def fused_linear_reshape_softmax_kernel(
    input_ptr,              # Input [19, 128] (flattened from [1, 19, 128])
    weight_ptr,             # Weight [128, 18] (transposed from [18, 128])
    bias_ptr,               # Bias [18]
    out_ptr,                # Output [19, 9, 1]
    M: tl.constexpr,        # Sequence length = 19
    K: tl.constexpr,        # Input features = 128
    N: tl.constexpr,        # Output features = 18
    softmax_dim: tl.constexpr,  # Softmax dimension size = 9
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Calculate program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute memory ranges
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, M)
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, N)
    
    # Initialize accumulator for linear transformation
    acc = tl.zeros((m_end - m_start, n_end - n_start), dtype=tl.float32)
    
    # Inner loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute ranges for current block
        k_block = min(BLOCK_SIZE_K, K - k)
        
        # Load input block: [m_block, k_block]
        input_block = tl.load(
            input_ptr + (m_start * K + k) + (m_end - m_start) * K + tl.arange(0, k_block)[None, :],
            mask=(tl.arange(0, k_block)[None, :] < k_block),
            other=0.0
        )
        
        # Load weight block: [k_block, n_block] (weight is already transposed)
        weight_block = tl.load(
            weight_ptr + (k * N + n_start) + k_block * N + tl.arange(0, n_end - n_start)[:, None],
            mask=(tl.arange(0, n_end - n_start)[:, None] < (n_end - n_start)),
            other=0.0
        )
        
        # Matrix multiplication: acc += input_block @ weight_block
        acc += tl.dot(input_block, weight_block)
    
    # Add bias to linear output
    bias_block = tl.load(
        bias_ptr + n_start,
        mask=tl.arange(0, n_end - n_start) < (n_end - n_start),
        other=0.0
    )
    acc = acc + bias_block[None, :]
    
    # Reshape and apply softmax
    # Original linear output: [m_block, n_block] = [m_block, features]
    # Reshape to: [m_block, softmax_dim, softmax_features] where softmax_dim=9 and softmax_features=2
    # Then apply softmax along dimension 1
    
    # Apply reshape: group features into pairs and take max for stability
    softmax_results = tl.zeros((m_end - m_start, softmax_dim), dtype=tl.float32)
    
    for i in range(acc.shape[0]):
        for j in range(softmax_dim):
            if j * 2 + 1 < acc.shape[1]:
                # Take max of two consecutive features for numerical stability
                max_val = tl.maximum(acc[i, j * 2], acc[i, j * 2 + 1])
                softmax_results[i, j] = max_val
            else:
                # Handle odd number of features (shouldn't happen with N=18)
                softmax_results[i, j] = acc[i, j * 2]
    
    # Apply softmax along dimension 1
    max_vals = tl.max(softmax_results, axis=1, keepdims=True)
    exp_vals = tl.exp(softmax_results - max_vals)
    sum_vals = tl.sum(exp_vals, axis=1, keepdims=True)
    softmax_results = exp_vals / sum_vals
    
    # Store final results: reshape from [m_block, softmax_dim] to [m_block, softmax_dim, 1]
    for i in range(softmax_results.shape[0]):
        for j in range(softmax_results.shape[1]):
            output_offset = ((m_start + i) * softmax_dim + j)
            tl.store(
                out_ptr + output_offset,
                softmax_results[i, j]
            )

@torch.fx.wrap
def fused_linear_reshape_softmax(in_0, in_1, in_2):
    M = 19  # Sequence length from [1, 19, 128] -> flattened to 19
    K = 128  # Input features  
    N = 18   # Output features
    softmax_dim = 9  # Softmax dimension size
    
    # Flatten input from [1, 19, 128] to [19, 128]
    input_flat = in_2.reshape([M, K])
    
    # Transpose weight from [18, 128] to [128, 18]
    weight_transposed = in_1.transpose(0, 1)
    
    # Create output tensor: [19, 9, 1]
    output_shape = (M, softmax_dim, 1)
    out = torch.empty(output_shape, dtype=torch.bfloat16, device=in_2.device)
    
    # Configure block sizes based on GPU optimization
    BLOCK_SIZE_M = 4
    BLOCK_SIZE_N = N  # Process all output features per block since 18 is small
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    num_M = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_N = (softmax_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel with proper tensor arguments
    fused_linear_reshape_softmax_kernel[(num_M, num_N)](
        input_flat,      # [19, 128] flattened input
        weight_transposed, # [128, 18] transposed weight
        in_0,            # [18] bias
        out.reshape([M, softmax_dim]),  # Reshape output to [19, 9] for kernel
        M, K, N, softmax_dim,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return out

# Replacement function - returns the fused kernel
def replacement_func():
    return fused_linear_reshape_softmax