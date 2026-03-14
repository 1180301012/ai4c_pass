import torch
import triton
import triton.language as tl


# Pattern matching function - matches the entire graph including all inputs
def pattern(in_0, in_1, in_2, in_3):
    # Include all inputs to avoid dead code
    # Match the linear + permute path
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
    tmp_3 = tmp_2.permute(0, 3, 1, 2)
    # Match the transpose path  
    tmp_4 = in_2.transpose(-2, -1)
    # Return as tuple to match model's return
    return tmp_3, tmp_4


# Extract arguments needed for replacement
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Optimized kernel for transpose using Triton
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
    ],
    key=['K'],
)
@triton.jit
def transpose_kernel(
    in_ptr, out_ptr,
    B, M, N, K,
    stride_in_b, stride_in_m, stride_in_n, stride_in_k,
    stride_out_b, stride_out_k, stride_out_n, stride_out_m,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Transpose: out[b, k, n, m] = in[b, m, n, k]
    Input: (B, M, N, K)
    Output: (B, K, N, M)
    """
    row_start = tl.program_id(0)
    row_offsets = row_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < (B * M * N * K)
    
    # Compute indices
    idx = row_offsets
    b_idx = idx // (M * N * K)
    idx = idx % (M * N * K)
    m_idx = idx // (N * K)
    idx = idx % (N * K)
    n_idx = idx // K
    k_idx = idx % K
    
    # Load input
    inp = tl.load(in_ptr + b_idx * stride_in_b + m_idx * stride_in_m + 
                  n_idx * stride_in_n + k_idx * stride_in_k,
                  mask=((b_idx < B) & (m_idx < M) & (n_idx < N) & (k_idx < K)), 
                  other=0.0)
    
    # Store output: out[b, k, n, m]
    out_ptrs = out_ptr + b_idx * stride_out_b + k_idx * stride_out_k + n_idx * stride_out_n + m_idx * stride_out_m
    tl.store(out_ptrs, inp, mask=mask)


# Optimized kernel for fused linear + permute using Triton
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 16, 'BLOCK_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 8, 'BLOCK_K': 1}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 4, 'BLOCK_K': 1}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 2, 'BLOCK_K': 1}, num_stages=4, num_warps=1),
    ],
    key=['M', 'K'],
)
@triton.jit
def fused_linear_permute_kernel(
    in_ptr, weight_ptr, bias_ptr, out_ptr,
    B, M, N, K, J,
    stride_in_b, stride_in_m, stride_in_k, stride_in_j,
    stride_w_n, stride_w_j,
    stride_out_b, stride_out_n, stride_out_m, stride_out_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused kernel: Linear + Permute"""
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_offs = tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_K), dtype=tl.float32)
    
    for j in range(J):
        in_ptrs = (pid_b * stride_in_b + m_offs[:, None, None] * stride_in_m + 
                   k_offs[None, None, :] * stride_in_k + j * stride_in_j)
        mask_in = (m_offs < M)[:, None, None] & (k_offs < K)[None, None, :]
        inp = tl.load(in_ptrs, mask=mask_in, other=0.0)
        
        w_ptrs = n_offs[:, None] * stride_w_n + j * stride_w_j
        mask_w = (n_offs < N)[:, None]
        weight = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc += inp * weight[None, :, None]
    
    bias = tl.load(bias_ptr + n_offs, mask=(n_offs < N), other=0.0)
    acc += bias[None, :, None]
    
    out_ptrs = (pid_b * stride_out_b + n_offs[:, None, None] * stride_out_n +
                m_offs[None, :, None] * stride_out_m + k_offs[None, None, :] * stride_out_k)
    mask_out = ((n_offs < N)[:, None, None] & (m_offs < M)[None, :, None] & 
                (k_offs < K)[None, None, :])
    tl.store(out_ptrs, acc, mask=mask_out)


@torch.fx.wrap
def transpose_triton_kernel_wrapper(in_2):
    """
    Compute transpose using Triton:
    Input: in_2 (B, M, N, K)
    Output: (B, K, N, M)
    """
    B, M, N, K = in_2.shape
    
    # Move to GPU if needed
    if in_2.device.type == 'cpu':
        in_2 = in_2.cuda()
    
    # Output
    out = torch.empty((B, K, N, M), dtype=in_2.dtype, device=in_2.device)
    
    # Grid: calculate number of blocks needed
    N_elements = B * M * N * K
    BLOCK_SIZE = 1024
    num_programs = (N_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    transpose_kernel[(num_programs,)](
        in_2, out,
        B, M, N, K,
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


@torch.fx.wrap
def fused_linear_permute_triton(in_3, in_1, in_0):
    """Compute linear + permute"""
    B, M, K, J = in_3.shape
    N = in_1.shape[0]
    
    if in_3.device.type == 'cpu':
        in_3 = in_3.cuda()
    if in_1.device.type == 'cpu':
        in_1 = in_1.cuda()
    if in_0.device.type == 'cpu':
        in_0 = in_0.cuda()
    
    out = torch.empty((B, N, M, K), dtype=in_3.dtype, device=in_3.device)
    
    grid = (B, M, K)
    
    fused_linear_permute_kernel[grid](
        in_3, in_1, in_0, out,
        B, M, N, K, J,
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        in_1.stride(0), in_1.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    
    return out


@torch.fx.wrap
def combined_kernel_wrapper(in_0, in_1, in_2, in_3):
    """Compute both outputs using optimized Triton kernels"""
    out1 = fused_linear_permute_triton(in_3, in_1, in_0)
    out2 = transpose_triton_kernel_wrapper(in_2)
    return out1, out2


def replacement_func():
    return combined_kernel_wrapper