import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the computation pattern:
    - tmp_0 = in_1 * scalar (0.125)
    - tmp_1 = in_0.transpose(-1, -2)
    Returns both results.
    """
    scalar = 0.125
    tmp_0 = in_1 * scalar
    tmp_1 = in_0.transpose(-1, -2)
    return (tmp_0, tmp_1)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_mul_transpose_kernel_125(
    in_0_ptr, in_1_ptr, out_0_ptr, out_1_ptr,
    M, N, K,
    stride_in_0_0, stride_in_0_1, stride_in_0_2, stride_in_0_3,
    stride_in_1_0, stride_in_1_1, stride_in_1_2, stride_in_1_3,
    stride_out_0_0, stride_out_0_1, stride_out_0_2, stride_out_0_3,
    stride_out_1_0, stride_out_1_1, stride_out_1_2, stride_out_1_3,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID for the out_0 computation (scalar multiplication)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_1 = num_pid_m * num_pid_n
    
    scalar = 0.125
    
    # Compute for scalar multiplication (in_1 * scalar)
    pid_in_1 = pid
    if pid_in_1 < num_pid_in_1:
        pid_m = pid_in_1 // num_pid_n
        pid_n = pid_in_1 % num_pid_n
        
        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        
        # Load from in_1: shape is [M, N, K, 8] or similar 4D tensor
        # We process the first two dimensions
        ptrs_in_1 = in_1_ptr + offs_m[:, None] * stride_in_1_0 + offs_n[None, :] * stride_in_1_1
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        
        # Load and multiply by scalar
        input_val = tl.load(ptrs_in_1, mask=mask, other=0.0)
        result = input_val * scalar
        
        # Store to out_0
        ptrs_out_0 = out_0_ptr + offs_m[:, None] * stride_out_0_0 + offs_n[None, :] * stride_out_0_1
        tl.store(ptrs_out_0, result, mask=mask)
    
    # Compute for transpose (swap last two dims)
    # in_0 shape: [*, *, K, 8] -> out_1 shape: [*, *, 8, K]
    total_programs = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(K, BLOCK_SIZE_N)
    pid_transpose = pid - num_pid_in_1
    
    if pid_transpose >= 0 and pid_transpose < total_programs:
        pid_m = pid_transpose // tl.cdiv(K, BLOCK_SIZE_N)
        pid_k = pid_transpose % tl.cdiv(K, BLOCK_SIZE_N)
        
        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_k = (pid_k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % K
        
        # Load from in_0: [M, N, K, 8], need to transpose last two dims -> [M, N, 8, K]
        ptrs_in_0 = in_0_ptr + offs_m[:, None] * stride_in_0_0 + offs_k[None, :] * stride_in_0_2
        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        
        input_trans = tl.load(ptrs_in_0, mask=mask, other=0.0)
        
        # Store to out_1 with transposed layout: [M, N, 8, K]
        ptrs_out_1 = out_1_ptr + offs_m[:, None] * stride_out_1_0 + offs_k[None, :] * stride_out_1_3
        tl.store(ptrs_out_1, input_trans, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper_125(in_0, in_1):
    """
    Fused kernel that performs:
    1. Scalar multiplication on in_1 (0.125)
    2. Transpose on in_0
    
    Returns both results as a tuple.
    """
    # Get input shapes
    M = in_0.shape[0]
    N = in_0.shape[1]
    K = in_0.shape[2]
    
    # Output for scalar multiplication (same shape as in_1)
    out_0 = torch.empty_like(in_1)
    
    # Output for transpose (swap last two dimensions)
    out_1_shape = list(in_0.shape)
    out_1_shape[-2], out_1_shape[-1] = out_1_shape[-1], out_1_shape[-2]
    out_1 = torch.empty(out_1_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Grid for scalar multiplication
    grid_m = triton.cdiv(M, 128)
    grid_n = triton.cdiv(N, 256)
    num_prods = grid_m * grid_n
    
    # Grid for transpose
    grid_trans_m = triton.cdiv(M, 128)
    grid_trans_k = triton.cdiv(K, 256)
    num_trans = grid_trans_m * grid_trans_k
    
    # Total grid size
    grid = (num_prods + num_trans,)
    
    fused_mul_transpose_kernel_125[grid](
        in_0, in_1, out_0, out_1,
        M, N, K,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        out_0.stride(0), out_0.stride(1), out_0.stride(2), out_0.stride(3),
        out_1.stride(0), out_1.stride(1), out_1.stride(2), out_1.stride(3),
    )
    
    return out_0, out_1


def replacement_func():
    return fused_kernel_wrapper_125