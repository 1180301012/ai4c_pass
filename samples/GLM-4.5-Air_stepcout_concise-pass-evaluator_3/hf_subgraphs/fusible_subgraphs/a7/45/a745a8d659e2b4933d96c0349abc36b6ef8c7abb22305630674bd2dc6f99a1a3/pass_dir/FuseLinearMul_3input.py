import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_mul_3input_kernel(
    input_ptr, weight_ptr, mul_input_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wk, stride_wn,
    stride_mm, stride_mn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused kernel: output = (input @ weight^T) * mul_input
    input shape: [M, K]
    weight shape: [N, K] - stored as [N, K], accessed as transposed [K, N]
    mul_input shape: [M, N]
    output shape: [M, N]
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

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Input: [M, K] - stride_im, stride_ik
    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    
    # Weight: [N, K] - stored as [N, K], but we need [K, N] for matmul
    # Access as: weight[k, n] where k iterates, n is fixed per block
    weight_ptrs = weight_ptr + (offs_n[:, None] * stride_wk + offs_k[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        input_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K)
        # For weight: iterate over K dimension (now first stride), pick N (second stride)
        weight_mask = (offs_k[None, :] < K - k * BLOCK_K) & (offs_n[:, None] < N)

        a = tl.load(input_ptrs, mask=input_mask, other=0.0)
        b = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        accumulator += tl.dot(a, b)

        input_ptrs += BLOCK_K * stride_im
        weight_ptrs += BLOCK_K * stride_wn
        offs_k += BLOCK_K

    accumulator = accumulator.to(tl.float16)

    # Load multiply input and apply element-wise multiply
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mul_ptrs = mul_input_ptr + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn
    mul_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    mul_input = tl.load(mul_ptrs, mask=mul_mask, other=0.0)
    accumulator = accumulator * mul_input

    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=output_mask)


def pattern(in_0, in_1, in_2):
    """Match the pattern: linear(in_1, in_0, None) followed by element-wise multiply with in_2"""
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.linear(in_1, tmp_0, None)
    tmp_0 = None
    tmp_2 = in_2 * tmp_1
    tmp_1 = None
    return (tmp_2,)


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments:
    - in_0: weight matrix [out_features, in_features]
    - in_1: input to linear [*, in_features]
    - in_2: input to multiply [*, out_features]
    """
    return (in_0, in_1, in_2)


@torch.fx.wrap
def fused_linear_mul_3input_kernel_wrapper(in_0, in_1, in_2):
    """
    Wrapper function that launches the fused linear + multiply kernel (3-input version).
    
    in_0: weight [out_features, in_features]
    in_1: linear_input [*, in_features]
    in_2: multiply_input [*, out_features]
    
    Returns: tuple of ((in_1 @ in_0^T) * in_2,)
    """
    # Handle batch dimensions - flatten all but last two dimensions
    original_shape = in_1.shape
    in_features = in_0.shape[1]
    out_features = in_0.shape[0]
    
    # Reshape inputs to 2D for the kernel
    if in_1.dim() > 2:
        batch_dims = original_shape[:-1]
        M = 1
        for dim in batch_dims:
            M *= dim
        in_1_2d = in_1.view(M, in_features)
        in_2_2d = in_2.view(M, out_features)
    else:
        in_1_2d = in_1
        in_2_2d = in_2
        M = in_1_2d.shape[0]
    
    N = out_features
    K = in_features
    
    # Allocate output
    output = torch.empty((M, N), dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel
    grid = (M * N,)
    
    fused_linear_mul_3input_kernel[grid](
        in_1_2d, in_0, in_2_2d, output,
        M, N, K,
        in_1_2d.stride(0), in_1_2d.stride(1),
        in_0.stride(1), in_0.stride(0),  # weight strides transposed
        in_2_2d.stride(0), in_2_2d.stride(1),
        output.stride(0), output.stride(1),
    )
    
    # Reshape output back to original batch shape
    output_reshaped = output.view(*original_shape[:-1], out_features)
    
    # Return the result as a tuple
    return (output_reshaped,)


def replacement_func():
    return fused_linear_mul_3input_kernel_wrapper