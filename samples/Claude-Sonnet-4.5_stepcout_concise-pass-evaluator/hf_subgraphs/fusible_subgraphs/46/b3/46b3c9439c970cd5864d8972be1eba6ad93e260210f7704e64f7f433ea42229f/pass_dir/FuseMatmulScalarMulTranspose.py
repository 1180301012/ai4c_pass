import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Pattern to match:
    - matmul: in_2 @ in_1
    - scalar multiply by in_0
    """
    tmp_0 = torch.matmul(in_2, in_1)
    tmp_1 = tmp_0 * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_matmul_scalar_mul_kernel_simple(
    # Pointers
    in_2_ptr, in_1_ptr, out_ptr,
    # Scalar
    scalar_val,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_in2_m, stride_in2_k,
    stride_in1_k, stride_in1_n,
    stride_out_m, stride_out_n,
    # Block size
    BLOCK_K: tl.constexpr,
):
    """
    Simple kernel for small matrices - computes one output element per program
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_m >= M or pid_n >= N:
        return
    
    # Accumulator
    acc = 0.0
    
    # Loop over K dimension with blocking
    for k_start in range(0, K, BLOCK_K):
        k_end = min(k_start + BLOCK_K, K)
        k_size = k_end - k_start
        
        # Load block from in_2[pid_m, k_start:k_end]
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_end
        
        in_2_ptrs = in_2_ptr + pid_m * stride_in2_m + offs_k * stride_in2_k
        a = tl.load(in_2_ptrs, mask=mask_k, other=0.0)
        
        # Load block from in_1[k_start:k_end, pid_n]
        in_1_ptrs = in_1_ptr + offs_k * stride_in1_k + pid_n * stride_in1_n
        b = tl.load(in_1_ptrs, mask=mask_k, other=0.0)
        
        # Accumulate dot product
        acc += tl.sum(a * b)
    
    # Multiply by scalar and store
    result = acc * scalar_val
    out_ptr_loc = out_ptr + pid_m * stride_out_m + pid_n * stride_out_n
    tl.store(out_ptr_loc, result)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_scalar_mul_kernel(
    # Pointers
    in_2_ptr, in_1_ptr, out_ptr,
    # Scalar
    scalar_val,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_in2_m, stride_in2_k,
    stride_in1_k, stride_in1_n,
    stride_out_m, stride_out_n,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel for: out = (in_2 @ in_1) * scalar
    in_2: [M, K]
    in_1: [K, N]
    out: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers for the first block
    in_2_ptrs = in_2_ptr + (offs_m[:, None] * stride_in2_m + offs_k[None, :] * stride_in2_k)
    in_1_ptrs = in_1_ptr + (offs_k[:, None] * stride_in1_k + offs_n[None, :] * stride_in1_n)
    
    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Matrix multiplication loop
    for k in range(0, K, BLOCK_SIZE_K):
        # Masks
        mask_in2 = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        mask_in1 = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        
        # Load
        a = tl.load(in_2_ptrs, mask=mask_in2, other=0.0)
        b = tl.load(in_1_ptrs, mask=mask_in1, other=0.0)
        
        # Matrix multiply
        acc += tl.dot(a, b)
        
        # Advance pointers
        in_2_ptrs += BLOCK_SIZE_K * stride_in2_k
        in_1_ptrs += BLOCK_SIZE_K * stride_in1_k
    
    # Multiply by scalar
    acc = acc * scalar_val
    
    # Store result
    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (offs_out_m[:, None] * stride_out_m + offs_out_n[None, :] * stride_out_n)
    mask_out = (offs_out_m[:, None] < M) & (offs_out_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


@torch.fx.wrap
def fused_matmul_scalar_mul(in_0, in_1, in_2):
    """
    Fused implementation of matmul + scalar multiply
    in_0: scalar
    in_1: [K, N]
    in_2: [M, K]
    
    Computes:
    - tmp = in_2 @ in_1  (shape: [M, N])
    - result = tmp * in_0
    
    Returns: result
    """
    M, K = in_2.shape
    K2, N = in_1.shape
    assert K == K2, f"Dimension mismatch: {K} != {K2}"
    
    # Allocate output
    out = torch.empty((M, N), device=in_2.device, dtype=in_2.dtype)
    
    # Extract scalar value
    scalar_val = in_0.item() if in_0.numel() == 1 else in_0
    
    # For small matrices, use simple kernel (one thread per output element)
    if M < 16 or N < 16:
        BLOCK_K = 128
        grid = (M, N)
        fused_matmul_scalar_mul_kernel_simple[grid](
            in_2, in_1, out,
            scalar_val,
            M, N, K,
            in_2.stride(0), in_2.stride(1),
            in_1.stride(0), in_1.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_K=BLOCK_K,
        )
    else:
        # For larger matrices, use tiled kernel with tl.dot
        grid = (
            triton.cdiv(M, 64),
            triton.cdiv(N, 64),
        )
        fused_matmul_scalar_mul_kernel[grid](
            in_2, in_1, out,
            scalar_val,
            M, N, K,
            in_2.stride(0), in_2.stride(1),
            in_1.stride(0), in_1.stride(1),
            out.stride(0), out.stride(1),
        )
    
    return out


def replacement_func():
    return fused_matmul_scalar_mul