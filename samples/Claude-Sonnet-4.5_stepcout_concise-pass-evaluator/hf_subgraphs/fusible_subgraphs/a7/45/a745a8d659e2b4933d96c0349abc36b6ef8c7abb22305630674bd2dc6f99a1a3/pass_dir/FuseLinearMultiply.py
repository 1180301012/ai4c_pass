import torch
import triton
import triton.language as tl


# Pattern matching function
def pattern(in_0, in_1, in_2):
    """
    Match: linear(in_1, in_0, None) * in_2
    where the intermediate linear result is not returned
    """
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.linear(in_1, tmp_0, None)
    tmp_2 = in_2 * tmp_1
    return (tmp_2,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        # Ultra-massive tiles for maximizing throughput on very large workloads
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        # More configs with 3 stages for better pipelining
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        # Configs with 4 stages
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        # Medium tiles
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        # Small M configs for handling edge cases
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_multiply_kernel(
    # Pointers to matrices
    in_1_ptr, in_0_ptr, in_2_ptr, out_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_in_1_m, stride_in_1_k,
    stride_in_0_n, stride_in_0_k,
    stride_in_2_m, stride_in_2_n,
    stride_out_m, stride_out_n,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused kernel: computes (in_1 @ in_0.T) * in_2
    in_1: [M, K]
    in_0: [N, K] (we transpose to get [K, N])
    in_2: [M, N]
    out: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers
    in_1_ptrs = in_1_ptr + offs_m[:, None] * stride_in_1_m + offs_k[None, :] * stride_in_1_k
    in_0_ptrs = in_0_ptr + offs_n[None, :] * stride_in_0_n + offs_k[:, None] * stride_in_0_k

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Matrix multiplication loop
    for k in range(0, K, BLOCK_K):
        # Load blocks
        in_1_block = tl.load(in_1_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
        in_0_block = tl.load(in_0_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        
        # Accumulate
        acc += tl.dot(in_1_block, in_0_block)
        
        # Advance pointers
        in_1_ptrs += BLOCK_K * stride_in_1_k
        in_0_ptrs += BLOCK_K * stride_in_0_k

    # Load the multiplication factor in_2
    in_2_ptrs = in_2_ptr + offs_m[:, None] * stride_in_2_m + offs_n[None, :] * stride_in_2_n
    in_2_block = tl.load(in_2_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)

    # Fused multiply
    out = acc.to(in_2_block.dtype) * in_2_block

    # Store result
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@torch.fx.wrap
def fused_linear_multiply(in_0, in_1, in_2):
    """
    Computes: (in_1 @ in_0.T) * in_2
    
    in_0: [N, K] - weight matrix
    in_1: [..., K] - input
    in_2: [..., N] - multiplier (gate)
    """
    # Get shapes
    original_shape = in_1.shape
    K = in_0.shape[1]
    N = in_0.shape[0]
    
    # Flatten input
    in_1_2d = in_1.reshape(-1, K)
    M = in_1_2d.shape[0]
    
    # Flatten in_2
    in_2_2d = in_2.reshape(M, N)
    
    # Allocate output
    out = torch.empty((M, N), dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    
    fused_linear_multiply_kernel[grid](
        in_1_2d, in_0, in_2_2d, out,
        M, N, K,
        in_1_2d.stride(0), in_1_2d.stride(1),
        in_0.stride(0), in_0.stride(1),
        in_2_2d.stride(0), in_2_2d.stride(1),
        out.stride(0), out.stride(1),
    )
    
    # Reshape output to match expected shape
    output_shape = list(original_shape[:-1]) + [N]
    out = out.reshape(output_shape)
    
    return out


def replacement_func():
    return fused_linear_multiply