import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def linear_3d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wk, stride_wn,
    stride_b,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Linear layer for 3D input (M, 1, K) -> (M, 1, N).
    We treat it as (M, K) @ (N, K).T + bias, then reshape to (M, 1, N).
    """
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

    # Input is (M, 1, K) - we view it as (M, K) for the gemm
    input_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    weight_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

    # Load input - treat 3D as 2D by squeezing the middle dim
    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    input_vals = tl.load(input_ptrs, mask=input_mask, other=0.0)

    weight_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
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


def linear_slice_3d_triton(input_tensor, weight, bias):
    """
    Fused linear + split operation for 3D input.
    input_tensor: (M, 1, K) - reshaped from (1, 150, 1, 512) -> (300, 1, 256)
    weight: (N, K) 
    bias: (N,)
    Returns: (M, 1, N/2), (M, 1, N/2)
    """
    # Input is (M, 1, K) - squeeze to (M, K) for computation
    M, one, K = input_tensor.shape
    assert one == 1, f"Expected second dim to be 1, got {one}"
    
    N = weight.shape[0]
    half_n = N // 2

    # Allocate full output (M, N)
    output_full = torch.empty((M, N), dtype=torch.float32, device=input_tensor.device)
    
    # Launch kernel for linear
    BLOCK_M = 32
    BLOCK_N = 256
    
    # Squeeze input to 2D for the kernel
    input_2d = input_tensor.squeeze(1)
    
    linear_3d_kernel[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)](
        input_2d, weight, bias, output_full,
        M, N, K,
        input_2d.stride(0), input_2d.stride(1),
        weight.stride(0), weight.stride(1),
        bias.stride(0),
        output_full.stride(0), output_full.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=triton.next_power_of_2(K)
    )

    # Split the output and reshape to 3D
    output_first = output_full[:, :half_n].unsqueeze(1)
    output_second = output_full[:, half_n:].unsqueeze(1)

    return output_first, output_second


@torch.fx.wrap
def linear_slice_3d_wrapper(in_4, in_3, in_2):
    """
    Wrapper for fused linear + slice.
    in_4: (1, 150, 1, 512) - will be reshaped to (300, 1, 256)
    in_3: (512, 256) - weight
    in_2: (512,) - bias
    Returns: (300, 1, 256), (300, 1, 256) - first half and second half
    """
    # Reshape in_4 from (1, 150, 1, 512) to (300, 1, 256)
    tmp_9 = in_4.reshape(300, -1, 256)
    return linear_slice_3d_triton(tmp_9, in_3, in_2)


def pattern(in_4, in_3, in_2):
    """
    Pattern: Reshape + Linear + slice into two halves.
    Returns the two halves of the linear output.
    """
    tmp_9 = in_4.reshape(300, -1, 256)
    tmp_10 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    tmp_11 = tmp_10[..., :256]
    tmp_12 = tmp_10[..., -256:]
    return tmp_11, tmp_12


def replacement_args(in_4, in_3, in_2):
    return (in_4, in_3, in_2)


def replacement_func():
    return linear_slice_3d_wrapper