import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    """Match Linear + Add + ReLU pattern"""
    linear = in_3 @ in_1.t() + in_0
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu()
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized kernel that fuses linear + add + ReLU operations
@triton.jit
def fused_linear_add_relu_kernel(
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
        # Handle M dimension in chunks for each K block
        for m in range(m_start, m_end):
            # Load input for this row
            x_offset = m * K + k
            x = tl.load(x_ptr + x_offset, mask=(m < M) & (k < K), other=0.0)
            x = x.to(tl.float32)
            
            # Load add input for this row (broadcast across N dimension)
            add_offset = m * N + n_start
            add_val = tl.load(add_input_ptr + add_offset, mask=(m < M) & (n_start < N), other=0.0)
            add_val = add_val.to(tl.float32)
            add_val = tl.broadcast_to(add_val, (n_end - n_start,))
            
            # Process N dimension for this M position
            for n in range(n_start, n_end):
                # Load weight for this output position
                weight_offset = n * K + k
                weight_val = tl.load(weight_ptr + weight_offset, mask=(n < N) & (k < K), other=0.0)
                weight_val = weight_val.to(tl.float32)
                
                # Load bias for this column
                bias = tl.load(bias_ptr + n, mask=(n < N), other=0.0)
                bias = bias.to(tl.float32)
                
                # Linear: x * weight + bias + add_input
                linear_val = x * weight_val + bias + add_val[n - n_start]
                
                # ReLU: max(0, linear_val)
                relu_val = tl.maximum(linear_val, 0.0)
                
                # Store in accumulator
                acc[m - m_start, n - n_start] = relu_val
        
        k = k_end
    
    # Store the result
    for m in range(m_start, m_end):
        for n in range(n_start, n_end):
            out_offset = m * N + n
            tl.store(out_ptr + out_offset, acc[m - m_start, n - n_start])

# Optimized kernel with better memory access pattern
@triton.jit
def fused_linear_add_relu_kernel_optimized(
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
    # Each program handles one output element
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Only work within bounds
    if pid_m >= M or pid_n >= N:
        return
    
    # Each program computes one output element: result[pid_m, pid_n]
    acc = 0.0
    
    # Vectorized dot product for better memory access
    for k in range(0, K, BLOCK_K):
        # Load a block of weights for output column pid_n
        k_end = min(k + BLOCK_K, K)
        weight_ptrs = weight_ptr + pid_n * K + k
        weights = tl.load(weight_ptrs, mask=k < K, other=0.0).to(tl.float32)
        
        # Load a block of inputs for row pid_m
        x_ptrs = x_ptr + pid_m * K + k
        inputs = tl.load(x_ptrs, mask=k < K, other=0.0).to(tl.float32)
        
        # Load bias and add input
        bias = tl.load(bias_ptr + pid_n, other=0.0).to(tl.float32)
        add_val = tl.load(add_input_ptr + pid_m * N + pid_n, other=0.0).to(tl.float32)
        
        # Accumulate linear part: sum(x * weight) + bias + add_input
        acc += tl.sum(inputs * weights) + bias + add_val
    
    # Apply ReLU
    acc = tl.maximum(acc, 0.0)
    
    # Store result
    tl.store(out_ptr + pid_m * N + pid_n, acc)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_linear_add_relu(in_0, in_1, in_2, in_3):
    M, N = in_2.shape  # Output shape from in_2: [1000, 128]
    K = in_3.shape[1]  # Input dim from in_3: [1000, 128]
    
    # Use smaller block sizes for better occupancy and compute specialization
    BLOCK_M = 4
    BLOCK_N = 4
    BLOCK_K = 32
    
    # Calculate grid size
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    # Create output tensor
    out = torch.empty((M, N), dtype=in_3.dtype, device=in_3.device)
    
    # Use the optimized kernel that computes one output element per program
    fused_linear_add_relu_kernel_optimized[(grid_m, grid_n)](
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
    return fused_linear_add_relu