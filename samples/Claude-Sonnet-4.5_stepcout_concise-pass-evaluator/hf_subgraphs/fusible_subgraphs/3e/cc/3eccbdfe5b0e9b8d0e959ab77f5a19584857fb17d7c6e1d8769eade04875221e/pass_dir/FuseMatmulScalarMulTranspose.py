import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    """
    Match the pattern: matmul -> scalar multiply
    in_2: [M, K], in_1: [K, N], in_0: scalar
    tmp_0 = matmul(in_2, in_1) -> [M, N]
    tmp_1 = tmp_0 * in_0 -> [M, N]
    """
    tmp_0 = torch.matmul(in_2, in_1)
    tmp_1 = tmp_0 * in_0
    return tmp_1

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Highly optimized kernel for small matmul + scalar multiply
# Specialized for cases where output is very small
@triton.jit
def fused_matmul_scalar_kernel(
    in_2_ptr,  # [M, K]
    in_1_ptr,  # [K, N]
    out_ptr,   # [M, N]
    scalar,    # scalar value
    M, K, N,
    stride_in2_m, stride_in2_k,
    stride_in1_k, stride_in1_n,
    stride_out_m, stride_out_n,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one output element [m, n]
    pid = tl.program_id(0)
    m = pid // N
    n = pid % N
    
    if m >= M or n >= N:
        return
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over K dimension with vectorized loads
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        
        # Load blocks
        in2_ptrs = in_2_ptr + m * stride_in2_m + k_offs * stride_in2_k
        in1_ptrs = in_1_ptr + k_offs * stride_in1_k + n * stride_in1_n
        
        mask = k_offs < K
        
        in2_val = tl.load(in2_ptrs, mask=mask, other=0.0)
        in1_val = tl.load(in1_ptrs, mask=mask, other=0.0)
        
        # Accumulate
        acc += tl.sum(in2_val * in1_val)
    
    # Multiply by scalar and store
    result = acc * scalar
    out_ptr_final = out_ptr + m * stride_out_m + n * stride_out_n
    tl.store(out_ptr_final, result)

# Kernel wrapper
@torch.fx.wrap
def fused_matmul_scalar(in_0, in_1, in_2):
    """
    Fused implementation of matmul + scalar multiply
    in_0: scalar
    in_1: [K, N]
    in_2: [M, K]
    Returns: out [M, N]
    """
    # Get dimensions
    M, K = in_2.shape
    K2, N = in_1.shape
    assert K == K2, f"Matmul dimension mismatch: {K} vs {K2}"
    
    # Allocate output
    out = torch.empty((M, N), device=in_2.device, dtype=in_2.dtype)
    
    # Extract scalar value
    if in_0.numel() == 1:
        scalar_val = in_0.item()
    else:
        scalar_val = in_0
    
    # Launch kernel: one program per output element
    grid = (M * N,)
    BLOCK_K = 512  # Process K dimension in blocks of 512
    
    fused_matmul_scalar_kernel[grid](
        in_2, in_1, out,
        scalar_val,
        M, K, N,
        in_2.stride(0), in_2.stride(1),
        in_1.stride(0), in_1.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=BLOCK_K,
    )
    
    # Return result
    return out

# Replacement function
def replacement_func():
    return fused_matmul_scalar