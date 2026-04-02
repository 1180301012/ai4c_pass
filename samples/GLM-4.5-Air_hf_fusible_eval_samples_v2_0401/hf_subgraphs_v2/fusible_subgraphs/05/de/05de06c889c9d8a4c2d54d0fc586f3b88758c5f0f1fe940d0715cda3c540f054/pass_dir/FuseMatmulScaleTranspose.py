import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern matching matrix multiplication + scalar multiplication"""
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    m: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused kernel for matmul + scale
    
    Args:
        in_0_ptr: pointer to scalar multiplier
        in_1_ptr: pointer to input matrix A (m, k)
        in_2_ptr: pointer to input matrix B (k, n) 
        out_ptr: pointer to output matrix (m, n) - scaled matmul result
        m, k, n: matrix dimensions
        BLOCK_SIZE_*: tile sizes
    """
    # Program ID for matrix multiplication
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Compute block range
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    # Compute memory offsets
    in_1_ptrs = in_1_ptr + rm[:, None] * k + rk[None, :]
    in_2_ptrs = in_2_ptr + rk[:, None] * n + rn[None, :]
    
    # Load scalar multiplier
    scale = tl.load(in_0_ptr)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension
    for k_idx in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        # Load blocks
        a = tl.load(in_1_ptrs, mask=(rm[:, None] < m) and (rk[None, :] < k), other=0.0)
        b = tl.load(in_2_ptrs, mask=(rk[:, None] < k) and (rn[None, :] < n), other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Update pointers for next k block
        in_1_ptrs += BLOCK_SIZE_K
        in_2_ptrs += BLOCK_SIZE_K * n
    
    # Apply scalar multiplication
    accumulator *= scale
    
    # Calculate output addresses
    out_ptrs = out_ptr + rm[:, None] * n + rn[None, :]
    
    # Store output
    tl.store(out_ptrs, accumulator, mask=(rm[:, None] < m) and (rn[None, :] < n))

@torch.fx.wrap
def fused_matmul_scale_transpose(in_0, in_1, in_2):
    """Fused implementation of matmul + scale, with transpose"""
    # Get tensor shapes
    m = in_2.shape[0]  # 2
    k = in_2.shape[1]  # 512
    n = in_1.shape[1]  # 1
    
    # Determine output shape
    out_shape = (m, n)  # (2, 1) - scaled matmul result
    
    # Create output tensor with same dtype as inputs
    out = torch.empty(out_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Block sizes - optimized for small matrices
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32  
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    grid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_size = grid_m * grid_n
    
    # Launch kernel
    fused_kernel[(grid_size,)](
        in_0_ptr=in_0,  # Pass scalar tensor as pointer
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        m=m,
        k=k,
        n=n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return fused_matmul_scale_transpose