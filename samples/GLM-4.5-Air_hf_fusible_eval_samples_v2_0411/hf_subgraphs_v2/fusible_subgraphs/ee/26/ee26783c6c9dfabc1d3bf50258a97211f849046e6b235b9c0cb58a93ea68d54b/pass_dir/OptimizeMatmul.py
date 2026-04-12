import torch
import triton
import triton.language as tl

def pattern(in_1, in_3):
    # Match the matrix multiplication operation
    result = in_1 @ in_3
    return result

def replacement_args(in_1, in_3):
    return (in_1, in_3)

@triton.jit
def optimized_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    
    pid_m = pid // (num_pid_n * num_pid_k)
    pid_n = (pid // num_pid_k) % num_pid_n
    pid_k = pid % num_pid_k
    
    # Compute memory offsets
    a_offset = pid_m * BLOCK_M * K + pid_k * BLOCK_K
    b_offset = pid_n * BLOCK_N + pid_k * BLOCK_K * N
    c_offset = pid_m * BLOCK_M * N + pid_n * BLOCK_N
    
    # Initialize accumulators
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    
    # Matrix multiplication loop
    for k in range(0, K, BLOCK_K):
        # Load blocks
        a_block = tl.load(a_ptr + a_offset + k * K,
                         (BLOCK_M, BLOCK_K),
                         mask=(k < K),
                         other=0.0)
        b_block = tl.load(b_ptr + b_offset + k * N,
                         (BLOCK_K, BLOCK_N),
                         mask=(k < K),
                         other=0.0)
        
        # Matrix multiply and accumulate
        accumulator += tl.dot(a_block, b_block)
    
    # Store result
    tl.store(c_ptr + c_offset, accumulator, (BLOCK_M, BLOCK_N))

@torch.fx.wrap
def optimized_matmul(in_1, in_3):
    # Handle 4D tensor [batch_size, num_heads, seq_len, head_dim]
    if len(in_1.shape) == 4:
        batch_size, num_heads, seq_len, head_dim = in_1.shape
        # Reshape to 2D for efficient matmul: [batch_size * num_heads * seq_len, head_dim]
        in_1_2d = in_1.reshape(batch_size * num_heads * seq_len, head_dim)
    else:
        # Use original shape for 2D tensors
        in_1_2d = in_1
        M_total, head_dim = in_1_2d.shape
    
    # Handle 2D tensor for in_3 [head_dim, seq_len_out]
    K2, N = in_3.shape
    # Verify dimensions match for matrix multiplication
    assert head_dim == K2, f"Dimension mismatch: {head_dim} != {K2}"
    
    # Define M_total based on input shape
    if len(in_1.shape) == 4:
        M_total = batch_size * num_heads * seq_len
    else:
        M_total = in_1_2d.shape[0]
    
    # Create output tensor
    output_2d = torch.empty((M_total, N), dtype=in_1.dtype, device=in_1.device)
    input_tensor = in_1_2d
    
    # Triton kernel launch configuration
    grid_size = (tl.cdiv(M_total, 128) * tl.cdiv(N, 128) * tl.cdiv(head_dim, 32),)
    
    # Launch optimized kernel
    optimized_matmul_kernel[grid_size](
        input_tensor,
        in_3,
        output_2d,
        M_total,
        N,
        head_dim,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
    )
    
    # Reshape back to original format if needed
    if len(in_1.shape) == 4:
        output = output_2d.reshape(batch_size, num_heads, seq_len, N)
    else:
        output = output_2d
    
    return output

def replacement_func():
    return optimized_matmul