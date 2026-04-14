import torch
import triton
import triton.language as tl

def pattern(linear_input, linear_weight, linear_bias, mul_tensor):
    """
    Pattern: Linear operation followed by transpose and element-wise multiplication.
    Current computation:
    linear = torch.nn.functional.linear(linear_input, linear_weight, linear_bias)  # [B, 768, 196]
    tmp_3 = linear.transpose(-1, -2)  # [B, 196, 768]  
    tmp_4 = mul_tensor * tmp_3  # [B, 196, 768]
    """

    linear = torch.nn.functional.linear(linear_input, linear_weight, linear_bias)
    tmp_3 = linear.transpose(-1, -2)
    tmp_4 = mul_tensor * tmp_3
    return tmp_4

def replacement_args(linear_input, linear_weight, linear_bias, mul_tensor):
    return (linear_input, linear_weight, linear_bias, mul_tensor)

@triton.jit
def transpose_elementwise_mul_kernel(
    input_ptr,    # Input tensor [B, 768, 196]
    mul_ptr,      # Multiplier tensor [B, 196, 768] 
    out_ptr,      # Output tensor [B, 196, 768]
    B: tl.constexpr,   # Batch size
    M: tl.constexpr,   # 196
    N: tl.constexpr,   # 768
    K: tl.constexpr,   # 196
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that fuses transpose and element-wise multiplication"""
    pid = tl.program_id(0)
    total_elements = B * M * N
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate positions in transposed space
    b = offsets // (M * N)
    remainder = offsets % (M * N)
    m = remainder // N  # 196 dim (was K, now transposed)
    n = remainder % N   # 768 dim (was M, now transposed)
    
    # Transpose access: map [B, N, K] -> [B, K, N] 
    # Here: map [B, 768, 196] -> [B, 196, 768]
    input_offset = b * N * K + n * K + m
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Load multiplier from [B, 196, 768] space
    mul_offset = b * (m + 1) * N + n  # Simplified access pattern
    mul_val = tl.load(mul_ptr + offsets, mask=mask, other=0.0)
    
    # Element-wise multiplication
    out_val = input_val * mul_val
    tl.store(out_ptr + offsets, out_val, mask=mask)

@triton.heuristics({
    "BLOCK_SIZE_M": lambda args: 128 if args["B"] >= 128 else 64,
    "BLOCK_SIZE_N": lambda args: 64 if args["B"] >= 128 else 32,
    "VECTOR_SIZE": lambda args: 4 if args["B"] >= 64 else 2,
})
@triton.jit
def linear_transpose_kernel(
    x_ptr,        # [B, 768, 196] - input tensor
    w_ptr,        # [196, 196] - weight tensor  
    b_ptr,        # [196] - bias tensor
    out_ptr,      # [B, 196, 768] - output (transposed result)
    B: tl.constexpr,
    M: tl.constexpr,  # 196
    N: tl.constexpr,  # 768
    K: tl.constexpr,  # 196
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
):
    """Optimized kernel with advanced vectorization and memory access patterns"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Output is [B, 196, 768] - we're computing directly in transposed space
    m_out = pid_m * BLOCK_SIZE_M  # 196 dim  
    n_out = pid_n * BLOCK_SIZE_N  # 768 dim
    
    m_end = min(m_out + BLOCK_SIZE_M, M)
    n_end = min(n_out + BLOCK_SIZE_N, N)
    
    # Optimized vectorized access
    offsets_k = tl.arange(0, VECTOR_SIZE)
    
    # Process each batch element with highly optimized memory access
    for b in range(B):
        # Initialize accumulator with optimized initialization
        acc = 0.0
        
        # Fast bias loading for in-bounds access
        if m_out < M:
            bias = tl.load(b_ptr + m_out, mask=(m_out < M), other=0.0)
            acc += bias
        
        # Highly vectorized matrix multiplication with stride optimization
        for k in range(0, K, VECTOR_SIZE):
            # Vectorized weight loading with optimized masking
            weight_offsets = k + offsets_k
            mask_weights = weight_offsets < K
            
            # Load weight vector with stride optimization
            weights = tl.load(w_ptr + (m_out * K + weight_offsets), 
                            mask=mask_weights & (m_out < M), other=0.0)
            
            # Corresponding input elements with stride optimization
            input_offsets = b * N * K + n_out * K + weight_offsets
            mask_inputs = (weight_offsets < K) & (n_out < N)
            
            # Vectorized input loading with memory coalescing
            inputs = tl.load(x_ptr + input_offsets, 
                           mask=mask_inputs, other=0.0)
            
            # Optimized vectorized dot product
            acc += tl.sum(inputs * weights)
        
        # Handle remaining elements when K is not divisible by VECTOR_SIZE
        remaining_start = (K // VECTOR_SIZE) * VECTOR_SIZE
        for kr in range(remaining_start, K):
            if (m_out < M) and (n_out < N):
                w_val = tl.load(w_ptr + (m_out * K + kr), mask=(kr < K) & (m_out < M), other=0.0)
                i_val = tl.load(x_ptr + (b * N * K + n_out * K + kr), mask=(kr < K) & (n_out < N), other=0.0)
                acc += w_val * i_val
        
        # Store result with optimized masking
        if (m_out < M) and (n_out < N):
            out_offset = b * M * N + m_out * N + n_out
            tl.store(out_ptr + out_offset, acc)

@triton.jit
def elementwise_mul_kernel(
    a_ptr,    # [B, 196, 768]
    b_ptr,    # [B, 196, 768]
    out_ptr,  # [B, 196, 768]
    B: tl.constexpr,
    M: tl.constexpr,  # 196
    N: tl.constexpr,  # 768
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise multiplication kernel"""
    total_elements = B * M * N
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out_vals = a_vals * b_vals
    
    tl.store(out_ptr + offsets, out_vals, mask=mask)

@torch.fx.wrap
def optimized_fused_operation(linear_input, linear_weight, linear_bias, mul_tensor):
    """Complete fused operation with advanced vectorization and optimization"""
    B = linear_input.shape[0]
    M, N, K = 196, 768, 196
    
    # Step 1: Compute linear + transpose with highly optimized vectorized kernels
    linear_transposed = torch.empty((B, M, N), dtype=linear_input.dtype, device=linear_input.device)
    
    # Advanced heuristics for vectorization and block sizing
    if B >= 128:
        block_m, block_n, vector_size = 128, 64, 4
    elif B >= 64:
        block_m, block_n, vector_size = 64, 64, 4
    elif B >= 32:
        block_m, block_n, vector_size = 64, 32, 2
    else:
        block_m, block_n, vector_size = 32, 32, 2
    
    grid_m = (M + block_m - 1) // block_m
    grid_n = (N + block_n - 1) // block_n
    
    # Execute the highly optimized kernel with advanced vectorization
    linear_transpose_kernel[(grid_m, grid_n)](
        linear_input, linear_weight, linear_bias, linear_transposed,
        B=B, M=M, N=N, K=K,
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, 
        VECTOR_SIZE=vector_size
    )
    
    # Step 2: Element-wise multiplication with optimized blocks
    result = torch.empty((B, M, N), dtype=linear_input.dtype, device=linear_input.device)
    
    # Dynamically adjust block size based on batch size for optimal GPU utilization
    BLOCK_SIZE = 8192 if B >= 128 else 4096 if B >= 64 else 2048
    total_elements = B * M * N
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    elementwise_mul_kernel[(num_programs,)](
        linear_transposed, mul_tensor, result,
        B=B, M=M, N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result

def replacement_func():
    return optimized_fused_operation