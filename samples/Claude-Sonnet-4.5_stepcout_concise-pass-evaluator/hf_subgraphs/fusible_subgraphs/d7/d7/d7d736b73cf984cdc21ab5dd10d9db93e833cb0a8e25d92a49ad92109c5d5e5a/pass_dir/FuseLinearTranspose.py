import torch
import triton
import triton.language as tl


def pattern(in_2, in_1, in_0):
    """
    Pattern to match: linear + transpose
    """
    tmp_2 = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3


def replacement_args(in_2, in_1, in_0):
    """
    Extract arguments for the replacement function
    """
    return (in_2, in_1, in_0)


@triton.autotune(
    configs=[
        # Extra-large blocks for maximum throughput  
        triton.Config({'BLOCK_M': 512, 'BLOCK_K': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 512, 'BLOCK_K': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
        # Large blocks with different configurations
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        # Medium blocks
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_transpose_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Matrix dimensions
    B, M, N, K,
    # Strides
    stride_ib, stride_im, stride_in,
    stride_wk, stride_wn,
    stride_ob, stride_ok, stride_om,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Fused linear + transpose kernel.
    
    Input: [B, M, N]
    Weight: [K, N]
    Bias: [K]
    Output: [B, K, M] (transposed result)
    
    Computation: output[b, k, m] = sum_n(input[b, m, n] * weight[k, n]) + bias[k]
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Determine batch, M block, and K block from flat program ID
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    
    pid_b = pid // (num_m_blocks * num_k_blocks)
    pid_mk = pid % (num_m_blocks * num_k_blocks)
    pid_m = pid_mk // num_k_blocks
    pid_k = pid_mk % num_k_blocks
    
    # Offsets for output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    
    # Initialize accumulator: [BLOCK_K, BLOCK_M]
    acc = tl.zeros((BLOCK_K, BLOCK_M), dtype=tl.float32)
    
    # Loop over N dimension
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        
        # Load input tile manually in transposed layout: [BLOCK_N, BLOCK_M]
        # We want input[b, m, n] arranged as [n, m] for efficient dot product
        input_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
        input_ptrs = input_ptr + pid_b * stride_ib + offs_n[:, None] * stride_in + offs_m[None, :] * stride_im
        input_tile_t = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Load weight tile: [BLOCK_K, BLOCK_N]
        weight_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        weight_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        weight_tile = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Accumulate: [BLOCK_K, BLOCK_M] += [BLOCK_K, BLOCK_N] @ [BLOCK_N, BLOCK_M]
        acc += tl.dot(weight_tile, input_tile_t)
    
    # Load and add bias: [BLOCK_K]
    bias_mask = offs_k < K
    bias_ptrs = bias_ptr + offs_k
    bias_tile = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    acc += bias_tile[:, None]
    
    # Store output: [B, K, M]
    output_mask = (offs_k[:, None] < K) & (offs_m[None, :] < M)
    output_ptrs = output_ptr + pid_b * stride_ob + offs_k[:, None] * stride_ok + offs_m[None, :] * stride_om
    tl.store(output_ptrs, acc.to(output_ptr.dtype.element_ty), mask=output_mask)


@torch.fx.wrap
def fused_linear_transpose(input, weight, bias):
    """
    Wrapper function for fused linear + transpose operation.
    
    Args:
        input: [B, M, N]
        weight: [K, N]
        bias: [K]
    
    Returns:
        output: [B, K, M]
    """
    B, M, N = input.shape
    K, N_w = weight.shape
    assert N == N_w, f"Dimension mismatch: input N={N}, weight N={N_w}"
    
    # Allocate output tensor with transposed shape
    output = torch.empty((B, K, M), dtype=input.dtype, device=input.device)
    
    # Grid configuration - use 1D grid for autotune compatibility
    def grid(META):
        return (
            B * triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(K, META['BLOCK_K']),
        )
    
    # Launch kernel
    fused_linear_transpose_kernel[grid](
        input, weight, bias, output,
        B, M, N, K,
        input.stride(0), input.stride(1), input.stride(2),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
    )
    
    return output


def replacement_func():
    """
    Return the replacement function (not called, just return the reference)
    """
    return fused_linear_transpose