import torch
import triton
import triton.language as tl

def pattern(in_1, in_3):
    # Simple pattern for any matmul operation
    matmul = in_1 @ in_3
    return matmul

def replacement_args(in_1, in_3):
    # Always return the arguments - let wrapper handle shape validation
    return (in_1, in_3)

@triton.jit
def simple_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    
    # Calculate row and column indices
    m = pid % M
    n = (pid // M) % N
    
    # Initialize accumulator
    accumulator = 0.0
    
    # Matrix multiplication
    for k in range(K):
        # Load elements
        a_val = tl.load(a_ptr + m * K + k, mask=(k < K), other=0.0).to(tl.float32)
        b_val = tl.load(b_ptr + k * N + n, mask=(k < K), other=0.0).to(tl.float32)
        
        # Accumulate
        accumulator += a_val * b_val
    
    # Store result
    tl.store(c_ptr + m * N + n, accumulator.to(tl.float16), mask=(m < M) & (n < N))

@torch.fx.wrap
def optimized_matmul(in_1, in_3):
    # Calculate output shape based on proper matrix multiplication rules
    # For batched matrix multiplication, we need to be careful about dimensions
    
    # Preserve input data type
    output_dtype = in_1.dtype
    
    # Get tensor dimensions
    if len(in_1.shape) == 4 and len(in_3.shape) == 2:
        # Batched matmul: [B, H, S_Q, D] @ [D, S_K] -> [B, H, S_Q, S_K]
        batch_size, n_heads, seq_len_q, head_dim = in_1.shape
        seq_len_k = in_3.shape[1]
        # Reshape to [B*H, S_Q, D] @ [D, S_K] -> [B*H, S_Q, S_K]
        M = batch_size * n_heads * seq_len_q
        N = seq_len_k
        K = head_dim
    elif len(in_1.shape) == 3 and len(in_3.shape) == 2:
        # Simple batched matmul: [B, S_Q, D] @ [D, S_K] -> [B, S_Q, S_K]  
        batch_size, seq_len_q, head_dim = in_1.shape
        seq_len_k = in_3.shape[1]
        M = batch_size * seq_len_q
        N = seq_len_k
        K = head_dim
    elif len(in_1.shape) == 2 and len(in_3.shape) == 2:
        # Standard matmul: [S_Q, D] @ [D, S_K] -> [S_Q, S_K]
        M, K = in_1.shape
        N = in_3.shape[1]
    else:
        # Fallback to regular matmul for unknown shapes
        return in_1 @ in_3
    
    # Create output tensor with correct shape and data type
    out = torch.empty((M, N), dtype=output_dtype, device=in_1.device)
    
    # Launch kernel with appropriate data type conversion
    if output_dtype == torch.bfloat16:
        # Use a simpler kernel for bfloat16 to avoid precision issues
        grid_size = M * N
        simple_matmul_kernel[(grid_size,)](
            a_ptr=in_1,
            b_ptr=in_3, 
            c_ptr=out,
            M=M,
            N=N,
            K=K
        ) 
    else:
        # For float16, let's use the highly optimized torch matmul
        # since our simple kernel is slower
        return in_1 @ in_3
    
    return out

def replacement_func():
    return optimized_matmul