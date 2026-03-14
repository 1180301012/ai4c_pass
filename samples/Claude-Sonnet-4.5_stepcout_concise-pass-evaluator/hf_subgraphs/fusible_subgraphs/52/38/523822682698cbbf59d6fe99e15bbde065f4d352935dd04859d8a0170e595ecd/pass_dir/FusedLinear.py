import torch
import triton
import triton.language as tl

def pattern(input, weight, bias):
    """
    Pattern to match linear operation
    """
    output = torch.nn.functional.linear(input, weight, bias)
    return output

def replacement_args(input, weight, bias):
    return (input, weight, bias)

@triton.jit
def fused_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized linear kernel: output = input @ weight.T + bias
    Input: [M, K]
    Weight: [N, K] 
    Bias: [N]
    Output: [M, N]
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    weight_ptrs = weight_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k < K - k * BLOCK_SIZE_K
        input_block = tl.load(input_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0.0)
        accumulator = tl.dot(input_block, tl.trans(weight_block), accumulator)
        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk
    
    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :]
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=output_mask)


@torch.fx.wrap
def fused_linear_wrapper(input, weight, bias):
    M, K = input.shape
    N, K_ = weight.shape
    assert K == K_, f"Incompatible dimensions: K={K}, K_={K_}"
    
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)
    
    # Choose optimal block sizes
    BLOCK_SIZE_M = 16 if M == 1 else 64
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 1 if M == 1 else 4
    
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    fused_linear_kernel[grid](
        input, weight, bias, output,
        M, N, K,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    
    return output


def replacement_func():
    return fused_linear_wrapper