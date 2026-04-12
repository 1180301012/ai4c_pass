import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation pattern
def pattern(in_0, in_1, in_2):
    """Pattern: matmul -> scalar multiplication -> transpose, returning both results"""
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    tmp_2 = tmp_1.T
    return (tmp_1, tmp_2)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2)

# Optimized Triton kernel that fuses matmul, scalar multiplication, and transpose
@triton.jit
def fused_matmul_scale_transpose_kernel(
    x_ptr,  # in_2: [2, 512]
    y_ptr,  # in_1: [512, 1] 
    scale,  # in_0: scalar (broadcast multiplier)
    out_ptr_2d,  # tmp_1: [2, 1] - original result
    out_ptr_1d,  # tmp_2: [1, 2] - transposed result
    M,  # rows in first matrix (2)
    N,  # cols in second matrix (1)
    K,  # inner dimension (512)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Fused kernel: matmul + scalar multiplication + efficient transpose"""
    
    # Program ID for parallel execution
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offset for matrix computation
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    k_offset = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator for matmul result with proper dtype
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=DTYPE)
    
    # Loop over K dimension for matmul
    for k in range(0, K, BLOCK_SIZE_K):
        # Bounds checking
        k_mask = k + k_offset < K
        m_mask = m_offset < M
        n_mask = n_offset < N
        
        # Load matrix slices
        x = tl.load(x_ptr + (m_offset[:, None] * K + k_mask[None, :]), 
                   mask=(m_mask[:, None] & k_mask[None, :]), other=0.0)
        y = tl.load(y_ptr + (k_mask[:, None] * N + n_offset[None, :]), 
                   mask=(k_mask[:, None] & n_mask[None, :]), other=0.0)
        
        # Matrix multiplication gemm with proper dtype
        accumulator = accumulator + tl.dot(x, y, acc_type=DTYPE)
    
    # Apply scalar multiplication
    accumulator = accumulator * scale
    
    # Store the original result [2, 1]
    out_mask_2d = (m_offset < M)[:, None] & (n_offset < N)[None, :]
    tl.store(out_ptr_2d + (m_offset[:, None] * N + n_offset[None, :]), 
             accumulator, mask=out_mask_2d)
    
    # For transpose [2, 1] -> [1, 2], we just need to swap the indices
    # m_offset becomes the column index (0-based 0 or 1)
    # n_offset becomes the row index (always 0 in this case since N=1)
    transposed_m = n_offset
    transposed_n = m_offset
    
    # Store the transposed result [1, 2] by reordering indices
    # [m, n] -> [n, m] for transpose
    out_mask_1d = (transposed_m < N)[:, None] & (transposed_n < M)[None, :]
    tl.store(out_ptr_1d + (transposed_m[:, None] * M + transposed_n[None, :]), 
             accumulator, mask=out_mask_1d)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_matmul_scale_transpose(in_0, in_1, in_2):
    """
    High-performance fusion of matmul, scalar multiplication, and transpose.
    Returns both the original result and its transpose.
    """
    # Get tensor shapes and dtypes
    M, K = in_2.shape  # [2, 512]
    N = in_1.shape[1]  # [512, 1] -> N = 1
    
    # Determine output shapes
    out_shape_2d = (M, N)  # [2, 1]
    out_shape_1d = (N, M)  # [1, 2] (transposed)
    
    # Create output tensors with the same dtype as inputs
    out_2d = torch.empty(out_shape_2d, dtype=in_2.dtype, device=in_2.device)
    out_1d = torch.empty(out_shape_1d, dtype=in_2.dtype, device=in_2.device)
    
    # Convert scalar to tensor for passing to kernel
    scale_tensor = torch.tensor([in_0.item()], dtype=in_2.dtype, device=in_2.device)
    
    # Map torch dtype to triton dtype
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    triton_dtype = dtype_map[in_2.dtype]
    
    # Optimized tile sizes for this specific problem
    BLOCK_SIZE_M = 2  # Process all rows at once since M=2
    BLOCK_SIZE_N = 1  # Process all columns at once since N=1  
    BLOCK_SIZE_K = 256  # Good for 512 dimension
    
    # Calculate grid size
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch the kernel
    fused_matmul_scale_transpose_kernel[(grid_m, grid_n)](
        x_ptr=in_2,
        y_ptr=in_1,
        scale=scale_tensor,  # Pass the scale tensor
        out_ptr_2d=out_2d,
        out_ptr_1d=out_1d,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        DTYPE=triton_dtype,
    )
    
    return (out_2d, out_1d)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_matmul_scale_transpose