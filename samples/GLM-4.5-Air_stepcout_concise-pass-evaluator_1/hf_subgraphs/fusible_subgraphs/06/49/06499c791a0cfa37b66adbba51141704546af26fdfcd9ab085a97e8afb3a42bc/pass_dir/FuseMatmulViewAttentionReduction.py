import torch
import triton
import triton.language as tl

# Pattern matching function for matmul operation only
def pattern(x, y):
    """
    Pattern: Simple matmul operation to debug pattern matching
    """
    z = torch.matmul(y, x)
    return (z,)

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Triton kernel for fused matmul + view operation
@triton.jit
def fused_matmul_view_kernel(
    # Input tensor A: [B, 1, C_out, K]
    a_ptr,
    # Input tensor B: [B, 1, K, 1] 
    b_ptr,
    # Output tensor: [B, C_out, 1, 1]
    out_ptr,
    # Strides for tensor A
    a_stride_batch,
    a_stride_channel, 
    a_stride_c_out,
    a_stride_k,
    # Strides for tensor B
    b_stride_batch,
    b_stride_channel,
    b_stride_k,
    b_stride_last,
    # Output strides  
    out_stride_batch,
    out_stride_c_out,
    # Tensor dimensions
    batch_size,
    c_out,
    k_dim,
    # Triton configuration
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for C_out dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for K dimension
):
    # Program identifiers for 2D launch grid
    pid_m = tl.program_id(0)  # Batch dimension
    pid_n = tl.program_id(1)  # C_out dimension
    
    # Calculate boundaries within program
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, batch_size)
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, c_out)
    
    # Initialize accumulator for this output element
    acc = 0.0
    
    # Loop over K dimension to compute dot product
    for k in range(0, k_dim, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, k_dim)
        
        # Calculate absolute indices
        m = min(pid_m * BLOCK_SIZE_M, batch_size - 1)
        n = min(pid_n * BLOCK_SIZE_N, c_out - 1) 
        k_idx = min(k, k_dim - 1)
        
        # Load elements from tensor A: [B, 1, C_out, K]
        # Address at position [m, 0, n, k_idx]
        a_addr = a_ptr + m * a_stride_batch + 0 * a_stride_channel + n * a_stride_c_out + k_idx * a_stride_k
        
        # Load elements from tensor B: [B, 1, K, 1] 
        # Address at position [m, 0, k_idx, 0]
        b_addr = b_ptr + m * b_stride_batch + 0 * b_stride_channel + k_idx * b_stride_k + 0 * b_stride_last
        
        a_val = tl.load(a_addr, mask=(k_idx < k_dim), other=0.0)
        b_val = tl.load(b_addr, mask=(k_idx < k_dim), other=0.0)
        
        # Matrix multiplication (dot product along K dimension)
        acc += a_val * b_val
    
    # Store the result at position [m, n, 0, 0]
    out_addr = out_ptr + m * out_stride_batch + n * out_stride_c_out + 0 + 0
    tl.store(out_addr, acc, mask=(m < batch_size) & (n < c_out))

# Triton kernel for reduction matmul: [B,1,C,K] @ [B,1,K,1] -> [B,1,C,1]
@triton.jit
def reduction_matmul_kernel(
    # Input tensor A: [B, 1, C_out, K]
    a_ptr,
    # Input tensor B: [B, 1, K, 1]
    b_ptr,
    # Output tensor: [B, 1, C_out, 1]
    out_ptr,
    # Strides for tensor A
    a_stride_batch,
    a_stride_channel,
    a_stride_c_out,
    a_stride_k,
    # Strides for tensor B
    b_stride_batch,
    b_stride_channel,
    b_stride_k,
    b_stride_last,
    # Output strides
    out_stride_batch,
    out_stride_channel,
    out_stride_c_out,
    out_stride_last,
    # Tensor dimensions
    batch_size,
    c_out,
    k_dim,
    # Triton configuration
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program identifiers for 2D launch grid
    pid_m = tl.program_id(0)  # Batch dimension
    pid_n = tl.program_id(1)  # C_out dimension
    
    # Calculate boundaries
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    m_end = min(m_offset + BLOCK_SIZE_M, batch_size)
    n_end = min(n_offset + BLOCK_SIZE_N, c_out)
    
    # Initialize accumulator for reduction along K dimension
    acc = 0.0
    
    # Loop over K dimension to compute dot product
    for k in range(0, k_dim, BLOCK_SIZE_K):
        k_idx = min(k, k_dim - 1)
        
        # Load from tensor A: [m, 0, n, k]
        a_addr = a_ptr + m_offset * a_stride_batch + 0 * a_stride_channel + n_offset * a_stride_c_out + k_idx * a_stride_k
        a_val = tl.load(a_addr, mask=(m_offset < batch_size) & (n_offset < c_out), other=0.0)
        
        # Load from tensor B: [m, 0, k, 0]  (always last dimension = 0)
        b_addr = b_ptr + m_offset * b_stride_batch + 0 * b_stride_channel + k_idx * b_stride_k + 0 * b_stride_last
        b_val = tl.load(b_addr, mask=(m_offset < batch_size), other=0.0)
        
        # Reduction along K dimension
        acc += a_val * b_val
    
    # Store result at [m, 0, n, 0]
    out_addr = out_ptr + m_offset * out_stride_batch + 0 * out_stride_channel + n_offset * out_stride_c_out + 0 * out_stride_last
    tl.store(out_addr, acc, mask=(m_offset < batch_size) & (n_offset < c_out))

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap  
def fused_matmul_wrapper(x, y):
    """
    Wrapper function that launches the reduction matmul kernel
    """
    # Get input tensor shapes
    B = y.shape[0]  # Batch size
    C_out = y.shape[2]  # Output channels  
    K = y.shape[3]  # Feature dimension
    
    # Create output tensor [B, 1, C_out, 1]
    output_shape = (B, 1, C_out, 1)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate strides
    if hasattr(x, 'stride'):
        x_stride_batch, x_stride_channel, x_stride_k, x_stride_last = x.stride()
    else:
        x_stride_batch = x.stride(0)
        x_stride_channel = x.stride(1)
        x_stride_k = x.stride(2)
        x_stride_last = x.stride(3)
    
    if hasattr(y, 'stride'):
        y_stride_batch, y_stride_channel, y_stride_c_out, y_stride_k = y.stride()
    else:
        y_stride_batch = y.stride(0)
        y_stride_channel = y.stride(1)
        y_stride_c_out = y.stride(2)
        y_stride_k = y.stride(3)
    
    out_stride_batch, out_stride_channel, out_stride_c_out, out_stride_last = out.stride()
    
    # Block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 256
    
    # Grid dimensions
    grid_m = (B + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (C_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Launch kernel
    reduction_matmul_kernel[grid](
        # Tensor pointers
        a_ptr=y,
        b_ptr=x,
        out_ptr=out,
        # Strides for tensor A
        a_stride_batch=y_stride_batch,
        a_stride_channel=y_stride_channel,
        a_stride_c_out=y_stride_c_out,
        a_stride_k=y_stride_k,
        # Strides for tensor B
        b_stride_batch=x_stride_batch,
        b_stride_channel=x_stride_channel,
        b_stride_k=x_stride_k,
        b_stride_last=x_stride_last,
        # Output strides
        out_stride_batch=out_stride_batch,
        out_stride_channel=out_stride_channel,
        out_stride_c_out=out_stride_c_out,
        out_stride_last=out_stride_last,
        # Dimensions
        batch_size=B,
        c_out=C_out,
        k_dim=K,
        # Triton config
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_matmul_wrapper