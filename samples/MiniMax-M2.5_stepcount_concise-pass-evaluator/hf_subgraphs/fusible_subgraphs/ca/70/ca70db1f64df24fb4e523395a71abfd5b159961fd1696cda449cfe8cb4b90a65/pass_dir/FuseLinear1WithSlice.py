import torch
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wk, stride_wn,
    stride_b,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    input_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    weight_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    input_vals = tl.load(input_ptrs, mask=input_mask, other=0.0)

    weight_ptrs = weight_ptr + (offs_n[:, None] * stride_wk + offs_k[None, :] * stride_wn)
    weight_vals = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

    gemm = tl.dot(input_vals, tl.trans(weight_vals))

    bias_ptrs = bias_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)

    output = gemm + bias_vals

    offs_m = offs_m[:, None]
    offs_n = offs_n[None, :]
    output_mask = (offs_m < M) & (offs_n < N)
    output_ptrs = output_ptr + (offs_m * stride_om + offs_n * stride_on)
    tl.store(output_ptrs, output, mask=output_mask)


def linear_slice_triton(input_tensor, weight, bias):
    """
    Fused linear operation.
    Handles both 2D (M, K) and 3D (M, 1, K) inputs.
    """
    # Handle both 2D and 3D inputs
    if input_tensor.dim() == 3:
        # 3D input: (M, 1, K) - squeeze to 2D
        input_2d = input_tensor.squeeze(1)
    else:
        input_2d = input_tensor
    
    M, K = input_2d.shape
    N = weight.shape[0]

    # Allocate full output
    output_full = torch.empty((M, N), dtype=torch.float32, device=input_tensor.device)
    
    # Launch kernel for linear with fixed block sizes
    linear_kernel[(triton.cdiv(M, 32) * triton.cdiv(N, 128),)](
        input_2d, weight, bias, output_full,
        M, N, K,
        input_2d.stride(0), input_2d.stride(1),
        weight.stride(0), weight.stride(1),
        bias.stride(0),
        output_full.stride(0), output_full.stride(1),
        BLOCK_M=32, BLOCK_N=128, BLOCK_K=256
    )

    return output_full


@torch.fx.wrap
def linear_slice_wrapper(in_5, in_1, in_0):
    """
    Wrapper for fused linear.
    in_5: (300, 256) - input
    in_1: (512, 256) - weight
    in_0: (512,) - bias
    Returns: (300, 512) - full linear output
    """
    return linear_slice_triton(in_5, in_1, in_0)


def pattern(in_5, in_1, in_0):
    """
    Pattern: Just the linear layer.
    This returns the linear output which then gets sliced in the model.
    """
    tmp_4 = torch.nn.functional.linear(in_5, in_1, in_0)
    return tmp_4


def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)


def replacement_func():
    return linear_slice_wrapper