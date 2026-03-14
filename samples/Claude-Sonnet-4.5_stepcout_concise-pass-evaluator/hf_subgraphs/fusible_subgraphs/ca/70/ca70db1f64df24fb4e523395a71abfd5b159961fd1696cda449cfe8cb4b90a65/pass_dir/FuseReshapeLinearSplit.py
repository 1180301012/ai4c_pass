import torch
import triton
import triton.language as tl

def pattern(in_5, in_1, in_0):
    """
    Pattern: linear -> slice -> view -> slice -> view -> unsqueeze
    """
    tmp_4 = torch.nn.functional.linear(in_5, in_1, in_0)
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 256, None)]
    tmp_6 = tmp_5.view(-1, 256)
    tmp_7 = tmp_4[slice(None, None, None), slice(-256, None, None)]
    tmp_8 = tmp_7.view(-1, 256)
    tmp_13 = tmp_6.unsqueeze(-2)
    return tmp_8, tmp_13

def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)

@triton.jit
def fused_linear_split_kernel(
    input_ptr, weight_ptr, bias_ptr,
    out_first_ptr, out_second_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused linear + split kernel
    Computes: output = input @ weight.T + bias, then splits into two halves
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matrix multiplication loop
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        
        # Load input block
        input_ptrs = input_ptr + offs_m[:, None] * stride_im + k_offs[None, :] * stride_ik
        input_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        input_vals = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Load weight block (transposed access)
        weight_ptrs = weight_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
        weight_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
        weight_vals = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(input_vals, tl.trans(weight_vals))
    
    # Add bias
    bias_ptrs = bias_ptr + offs_n
    bias_mask = offs_n < N
    bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    acc += bias_vals[None, :]
    
    # Determine which half to write to
    half_size = N // 2
    
    # Write to appropriate output
    offs_out = offs_n
    mask = (offs_m[:, None] < M) & (offs_out[None, :] < N)
    
    if pid_n * BLOCK_N < half_size:
        # First half
        out_ptrs = out_first_ptr + offs_m[:, None] * half_size + offs_out[None, :]
        out_mask = (offs_m[:, None] < M) & (offs_out[None, :] < half_size)
        tl.store(out_ptrs, acc, mask=out_mask)
    else:
        # Second half
        offs_out_shifted = offs_out - half_size
        out_ptrs = out_second_ptr + offs_m[:, None] * half_size + offs_out_shifted[None, :]
        out_mask = (offs_m[:, None] < M) & (offs_out_shifted[None, :] >= 0) & (offs_out_shifted[None, :] < half_size)
        tl.store(out_ptrs, acc, mask=out_mask)

@torch.fx.wrap
def fused_linear_split_reshape_unsqueeze(in_5, in_1, in_0):
    """
    Optimized implementation using Triton kernels
    Computes linear, splits, reshapes, and unsqueezes first half
    """
    # in_5: [300, 256], in_1: [512, 256], in_0: [512]
    M, K = in_5.shape
    N, K_w = in_1.shape
    
    # Output: [300, 512] split into two [300, 256]
    out_first = torch.empty((M, N // 2), device=in_5.device, dtype=in_5.dtype)
    out_second = torch.empty((M, N // 2), device=in_5.device, dtype=in_5.dtype)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    fused_linear_split_kernel[grid](
        in_5, in_1, in_0,
        out_first, out_second,
        M, N, K,
        in_5.stride(0), in_5.stride(1),
        in_1.stride(0), in_1.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    # tmp_8 = second half, already [300, 256]
    # tmp_13 = first half unsqueezed to [300, 1, 256]
    tmp_8 = out_second
    tmp_13 = out_first.unsqueeze(-2)
    
    return tmp_8, tmp_13

def replacement_func():
    return fused_linear_split_reshape_unsqueeze