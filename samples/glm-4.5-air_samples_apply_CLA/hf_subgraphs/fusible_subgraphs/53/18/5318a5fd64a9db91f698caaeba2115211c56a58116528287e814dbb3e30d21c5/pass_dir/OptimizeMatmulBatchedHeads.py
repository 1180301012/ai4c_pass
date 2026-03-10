import torch
import triton
import triton.language as tl

# Pattern matching function for batched matrix multiplication with heads
def pattern(in_1, in_0):
    tmp_0 = in_1 @ in_0
    return tmp_0

# Argument extraction function
def replacement_args(in_1, in_0):
    return (in_1, in_0)

# Optimized Triton kernel for batched matrix multiplication with multiple heads
@triton.jit
def batched_matmul_kernel(
    a_ptr, 
    b_ptr, 
    c_ptr, 
    batch_size,
    num_heads,
    m,       # sequence length dimension
    k,       # head dimension (shared)
    n,       # output head dimension
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Get program ID (3D grid only)
    pid_batch_head = tl.program_id(0)  # Combined batch and head
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Extract batch and head from combined ID
    pid_batch = pid_batch_head // num_heads
    pid_head = pid_batch_head % num_heads
    
    # Compute memory address offsets
    batch_offset = pid_batch * num_heads * m * k  # Use k instead of n for A
    head_offset = pid_head * m * k
    
    # Create offsets for the block
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Broadcast offsets for matrix multiplication: A is [m, k], B is [k, n]
    # A: batch + head + (m, k) = batch_offset + head_offset + m_offsets * k + k_offsets
    # B: batch + head + (k, n) = batch_offset + head_offset + k_offsets * n + n_offsets
    a_offsets = (batch_offset + head_offset + 
                 m_offsets[:, None] * k + k_offsets[None, :])
    b_offsets = (batch_offset + head_offset + 
                 k_offsets[:, None] * n + n_offsets[None, :])
    c_offsets = (batch_offset + head_offset +
                 m_offsets[:, None] * n + n_offsets[None, :])
    
    # Create masks to handle boundaries
    a_mask = (m_offsets[:, None] < m) & (k_offsets[None, :] < k)
    b_mask = (k_offsets[:, None] < k) & (n_offsets[None, :] < n)
    c_mask = (m_offsets[:, None] < m) & (n_offsets[None, :] < n)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (tiled)
    for k_val in range(0, k, BLOCK_SIZE_K):
        # Load A and B tiles correctly shaped for dot product
        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Matrix multiply: a (m, k) @ b (k, n) -> acc (m, n)
        accumulator += tl.dot(a, b)
        
        # Update offsets for next iteration
        k_offsets += BLOCK_SIZE_K
        # Update A and B offsets for next K block
        a_offsets = (batch_offset + head_offset + 
                     m_offsets[:, None] * k + k_offsets[None, :])
        b_offsets = (batch_offset + head_offset + 
                     k_offsets[:, None] * n + n_offsets[None, :])
    
    # Store result
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def batched_matmul_triton(in_1, in_0):
    # Get tensor shapes
    batch_size, num_heads, seq_len, head_dim = in_1.shape
    _, _, in_0_head_dim, out_head_dim = in_0.shape
    
    # Output shape
    out_shape = [batch_size, num_heads, seq_len, out_head_dim]
    output = torch.empty(out_shape, dtype=torch.float32, device=in_1.device)
    
    # Set up grid dimensions (3D grid)
    total_batch_head = batch_size * num_heads  # Combine batch and head
    m_grid_size = (seq_len + 63) // 64  # BLOCK_SIZE_M = 64
    n_grid_size = (out_head_dim + 63) // 64  # BLOCK_SIZE_N = 64
    
    # Launch kernel with 3D grid
    batched_matmul_kernel[(total_batch_head, m_grid_size, n_grid_size)](
        in_1,
        in_0,
        output,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        out_head_dim,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return batched_matmul_triton