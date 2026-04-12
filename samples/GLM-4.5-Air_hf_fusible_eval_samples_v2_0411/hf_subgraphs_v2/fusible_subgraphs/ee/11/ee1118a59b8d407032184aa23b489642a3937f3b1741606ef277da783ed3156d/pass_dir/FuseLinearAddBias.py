import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    """Match Linear + Add pattern"""
    linear = in_3 @ in_1.t() + in_0
    tmp_3 = in_2 + linear
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized kernel that fuses linear + add operations
@triton.jit
def fused_linear_add_kernel(
    x_ptr,           # in_3: input tensor [1000, 128]
    weight_ptr,      # in_1: weight tensor [128, 128] 
    bias_ptr,        # in_0: bias tensor [128]
    add_input_ptr,   # in_2: tensor to add [1000, 128]
    out_ptr,         # output [1000, 128]
    M,               # batch size = 1000
    N,               # hidden dim = 128
    K,               # input dim = 128
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one block of the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block bounds
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    k_start = 0
    m_end = min((pid_m + 1) * BLOCK_M, M)
    n_end = min((pid_n + 1) * BLOCK_N, N)
    
    # Initialize accumulator for the entire block
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Process K dimension in blocks
    k = k_start
    while k < K:
        # Bounds for this K block
        k_end = min(k + BLOCK_K, K)
        # Load weight block (transpose for better memory access)
        weight = tl.load(weight_ptr + n_start * K + k, mask=(n_start < N) & (k < K), other=0.0)
        weight = weight.to(tl.float32)
        
        # Handle M dimension in chunks for each K block
        for m in range(m_start, m_end):
            # Load input for this row
            x_offset = m * K + k
            x = tl.load(x_ptr + x_offset, mask=(m < M) & (k < K), other=0.0)
            x = x.to(tl.float32)
            
            # Load bias for this column (broadcast)
            bias = tl.load(bias_ptr + n_start, mask=(n_start < N), other=0.0)
            bias = bias.to(tl.float32)
            
            # Load add input for this output position
            add_offset = m * N + n_start
            add_val = tl.load(add_input_ptr + add_offset, mask=(m < M) & (n_start < N), other=0.0)
            add_val = add_val.to(tl.float32)
            
            # Accumulate for this row
            for n in range(n_start, n_end):
                weight_offset = n * K + k
                weight_val = tl.load(weight_ptr + weight_offset, mask=(n < N) & (k < K), other=0.0)
                weight_val = weight_val.to(tl.float32)
                
                # Linear: x * weight + bias
                linear_val = x * weight_val + bias
                # Add: linear_val + add_input
                result = linear_val + add_val
                
                # Store in accumulator
                acc[m - m_start, n - n_start] = result
        
        k = k_end
    
    # Store the result
    for m in range(m_start, m_end):
        for n in range(n_start, n_end):
            out_offset = m * N + n
            tl.store(out_ptr + out_offset, acc[m - m_start, n - n_start])

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_linear_add(in_0, in_1, in_2, in_3):
    M, N = in_2.shape  # Output shape from in_2: [1000, 128]
    K = in_3.shape[1]  # Input dim from in_3: [1000, 128]
    
    # Optimal tile sizes for this compute pattern
    BLOCK_M = 32
    BLOCK_N = 32  
    BLOCK_K = 32
    
    # Calculate grid size
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    # Create output tensor
    out = torch.empty((M, N), dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel
    fused_linear_add_kernel[(grid_m, grid_n)](
        x_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        add_input_ptr=in_2,
        out_ptr=out,
        M=M,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_linear_add