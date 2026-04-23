import torch
import triton
import triton.language as tl


@triton.jit
def fused_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel that computes linear (matmul + bias) operation.
    Automatically handles dtype based on output tensor type.
    """
    pid = tl.program_id(0)
    
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    if pid_m * BLOCK_SIZE_M >= M or pid_n * BLOCK_SIZE_N >= N:
        return
    
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    input_ptrs = input_ptr + (offs_m[:, None] * K + offs_k[None, :])
    weight_ptrs = weight_ptr + (offs_k[:, None] * N + offs_n[None, :])
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(K, 0, -BLOCK_SIZE_K):
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask_k = offs_k < K
        mask = mask_m[:, None] & mask_n[None, :] & mask_k[None, :]
        
        a = tl.load(input_ptrs, mask=mask, other=0.0)
        b = tl.load(weight_ptrs, mask=mask, other=0.0)
        
        acc += tl.dot(a, b)
        input_ptrs += BLOCK_SIZE_K
        weight_ptrs += BLOCK_SIZE_K * N
        offs_k = offs_k + BLOCK_SIZE_K
    
    if bias_ptr is not None:
        offs_n_bias = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n_bias = offs_n_bias < N
        bias = tl.load(bias_ptr + offs_n, mask=mask_n_bias, other=0.0)
        acc = acc + bias[None, :]
    
    offs_m_out = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_out = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m_out = offs_m_out < M
    mask_n_out = offs_n_out < N
    
    output_ptrs = output_ptr + (offs_m_out[:, None] * N + offs_n_out[None, :])
    # Cast output based on the output tensor's dtype
    output_dtype = output_ptr.dtype.element_ty
    tl.store(output_ptrs, acc.to(output_dtype), mask=(mask_m_out[:, None] & mask_n_out[None, :]))


def fused_linear_impl(input, weight, bias):
    """
    Implementation for fused linear operation.
    Handles both 2D and 3D inputs by flattening to 2D for the matmul.
    Automatically handles dtype based on input.
    """
    original_shape = input.shape
    if input.dim() == 3:
        input = input.reshape(-1, input.shape[-1])
    
    M = input.shape[0]
    N = weight.shape[0]
    K = weight.shape[1]
    
    output_dtype = input.dtype
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_programs = grid_m * grid_n
    num_warps = 4
    
    output = torch.empty((M, N), dtype=output_dtype, device=input.device)
    
    fused_linear_kernel[(num_programs,)](
        input, weight, bias, output,
        M, N, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        num_warps=num_warps,
    )
    
    if len(original_shape) == 3:
        output = output.reshape(original_shape[0], original_shape[1], N)
    
    return output


@torch.fx.wrap
def fused_linear_wrapper(input, weight, bias):
    """
    Wrapper function for fused linear operation.
    Automatically handles dtype based on input.
    """
    return fused_linear_impl(input, weight, bias)


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: dropout + linear
    For BigBird: dropout(in_2, 0.1, False, False) + linear(in_1, in_0)
    When dropout has no effect (training=False), we can fuse it with linear.
    """
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear


def pattern_with_cast(in_0, in_1, in_2):
    """
    Match the pattern: dropout + to(dtype) + linear
    For RECT_L: dropout(in_2, p=0.0, training=False) + to(dtype) + linear(in_1, in_0)
    When dtype is same as input, the cast is no-op and can be fused.
    """
    tmp_2 = torch.nn.functional.dropout(in_2, p=0.0, training=False)
    to = tmp_2.to(torch.float16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_linear_wrapper