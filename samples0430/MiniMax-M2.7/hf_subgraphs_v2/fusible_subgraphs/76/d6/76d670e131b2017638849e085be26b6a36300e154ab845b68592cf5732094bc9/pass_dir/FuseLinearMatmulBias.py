import torch
import triton
import triton.language as tl


# Define block sizes as constexpr
BLOCK_M = tl.constexpr(16)
BLOCK_N = tl.constexpr(16)
BLOCK_K = tl.constexpr(32)


@triton.jit
def triton_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
):
    """
    Triton kernel for fused linear: y = x @ W^T + b
    
    Layout:
    - input: [M, K] with strides (stride_im, stride_ik)
    - weight: [N, K] with strides (stride_wn, stride_wk) 
    - output: [M, N]
    """
    pid = tl.program_id(0)
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = num_pid_m
    if M - first_pid_m < group_size_m:
        group_size_m = M - first_pid_m
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Input pointers: access input[m, k] - shape (BLOCK_M, BLOCK_K)
    inp_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    mask_inp = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    inp = tl.load(inp_ptrs, mask=mask_inp, other=0.0)
    
    # Weight pointers: access weight^T[k, n] = weight.T[n, k] - shape (BLOCK_K, BLOCK_N)
    # For weight.T[k, n], offset = k * stride_wn + n * stride_wk
    wgt_ptrs = weight_ptr + (offs_k[:, None] * stride_wn + offs_n[None, :] * stride_wk)
    mask_wgt = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    wgt = tl.load(wgt_ptrs, mask=mask_wgt, other=0.0)
    
    acc = tl.dot(inp, wgt)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    
    out_ptrs = output_ptr + (offs_m[:, None] * N + offs_n[None, :])
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


@torch.fx.wrap
def triton_linear(input, weight, bias):
    """
    Wrapper for the Triton linear kernel.
    input: [batch, in_features]
    weight: [out_features, in_features] 
    bias: [out_features]
    output: [batch, out_features]
    """
    M, K = input.shape
    N = weight.shape[0]
    
    # Allocate output
    output = torch.empty((M, N), dtype=input.dtype, device=input.device)
    
    # Calculate grid
    grid_m = (M + 16 - 1) // 16
    grid_n = (N + 16 - 1) // 16
    
    grid = (grid_m * grid_n,)
    
    # Launch kernel
    triton_linear_kernel[grid](
        input, weight, bias, output,
        M, N, K,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1)
    )
    
    return output


def pattern(in_6, in_5, in_4):
    """
    Pattern for torch.nn.functional.linear
    F.linear(input, weight, bias) = input @ weight.T + bias
    """
    linear = torch.nn.functional.linear(in_6, in_5, in_4)
    return linear


def replacement_args(in_6, in_5, in_4):
    return (in_6, in_5, in_4)


def replacement_func():
    return triton_linear