import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Full pattern - must match EXACT syntax from model.py including aliases
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = torch.nn.functional.linear(in_5, tmp_1, tmp_0)
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 256, None)]
    tmp_6 = tmp_5.view(-1, 256)
    tmp_7 = tmp_4[slice(None, None, None), slice(-256, None, None)]
    tmp_8 = tmp_7.view(-1, 256)
    tmp_9 = in_4.reshape(300, -1, 256)
    tmp_10 = torch.nn.functional.linear(tmp_9, tmp_3, tmp_2)
    tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]
    tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]
    tmp_13 = tmp_6.unsqueeze(-2)
    return (tmp_11, tmp_12, tmp_8, tmp_13)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

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
    if pid_n * BLOCK_N < half_size:
        # First half
        out_ptrs = out_first_ptr + offs_m[:, None] * half_size + offs_n[None, :]
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < half_size)
        tl.store(out_ptrs, acc, mask=out_mask)
    else:
        # Second half
        offs_n_shifted = offs_n - half_size
        out_ptrs = out_second_ptr + offs_m[:, None] * half_size + offs_n_shifted[None, :]
        out_mask = (offs_m[:, None] < M) & (offs_n_shifted[None, :] >= 0) & (offs_n_shifted[None, :] < half_size)
        tl.store(out_ptrs, acc, mask=out_mask)

@torch.fx.wrap
def fused_full_computation(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Optimized implementation using Triton kernels
    """
    # First linear operation: in_5 @ in_1.T + in_0
    M1, K1 = in_5.shape
    N1, K1_w = in_1.shape
    
    out1_first = torch.empty((M1, N1 // 2), device=in_5.device, dtype=in_5.dtype)
    out1_second = torch.empty((M1, N1 // 2), device=in_5.device, dtype=in_5.dtype)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    grid1 = (triton.cdiv(M1, BLOCK_M), triton.cdiv(N1, BLOCK_N))
    
    fused_linear_split_kernel[grid1](
        in_5, in_1, in_0,
        out1_first, out1_second,
        M1, N1, K1,
        in_5.stride(0), in_5.stride(1),
        in_1.stride(0), in_1.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    # Second linear operation
    tmp_9 = in_4.reshape(300, -1, 256)
    B, M2, K2 = tmp_9.shape
    N2, K2_w = in_3.shape
    
    tmp_9_flat = tmp_9.reshape(B * M2, K2)
    
    out2_first_flat = torch.empty((B * M2, N2 // 2), device=in_4.device, dtype=in_4.dtype)
    out2_second_flat = torch.empty((B * M2, N2 // 2), device=in_4.device, dtype=in_4.dtype)
    
    grid2 = (triton.cdiv(B * M2, BLOCK_M), triton.cdiv(N2, BLOCK_N))
    
    fused_linear_split_kernel[grid2](
        tmp_9_flat, in_3, in_2,
        out2_first_flat, out2_second_flat,
        B * M2, N2, K2,
        tmp_9_flat.stride(0), tmp_9_flat.stride(1),
        in_3.stride(0), in_3.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    tmp_11 = out2_first_flat.reshape(B, M2, N2 // 2)
    tmp_12 = out2_second_flat.reshape(B, M2, N2 // 2)
    tmp_8 = out1_second
    tmp_13 = out1_first.unsqueeze(-2)
    
    return tmp_11, tmp_12, tmp_8, tmp_13

def replacement_func():
    return fused_full_computation