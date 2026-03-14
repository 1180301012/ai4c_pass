import torch
import triton
import triton.language as tl


def pattern(bias, weight, input_tensor, residual):
    """
    Pattern: Linear -> Dropout -> Add (residual connection)
    Since dropout with training=False is a no-op, we can fuse this into a single operation.
    """
    linear_out = torch.nn.functional.linear(input_tensor, weight, bias)
    dropout_out = torch.nn.functional.dropout(linear_out, p=0.0, training=False)
    result = residual + dropout_out
    return result


def replacement_args(bias, weight, input_tensor, residual):
    return (input_tensor, weight, bias, residual)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_add_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, bias_ptr, residual_ptr, output_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_rm, stride_rn,
    stride_om, stride_on,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Fused kernel for: output = (input @ weight.T) + bias + residual
    - input: (M, K)
    - weight: (N, K)
    - bias: (N,)
    - residual: (M, N)
    - output: (M, N)
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create offset ranges
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize pointers for input and weight
    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    weight_ptrs = weight_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    # Accumulator for matrix multiplication
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Matrix multiplication loop
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks
        input_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        weight_mask = (offs_n[:, None] < N) & ((k + offs_k[None, :]) < K)
        
        a = tl.load(input_ptrs, mask=input_mask, other=0.0)
        b = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Accumulate
        accumulator += tl.dot(a, tl.trans(b))
        
        # Advance pointers
        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    # Load bias
    bias_ptrs = bias_ptr + offs_n
    bias_mask = offs_n < N
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    
    # Add bias (broadcast across M dimension)
    accumulator += bias[None, :]

    # Load residual and add
    residual_ptrs = residual_ptr + (offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn)
    residual_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    residual = tl.load(residual_ptrs, mask=residual_mask, other=0.0)
    accumulator += residual

    # Store output
    output_ptrs = output_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=output_mask)


@torch.fx.wrap
def fused_linear_add(input_tensor, weight, bias, residual):
    """
    Wrapper function for the fused linear + add kernel.
    Computes: output = (input @ weight.T) + bias + residual
    """
    # Handle multi-dimensional inputs by reshaping
    original_shape = input_tensor.shape
    if input_tensor.dim() > 2:
        input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])
        residual = residual.reshape(-1, residual.shape[-1])
    
    M, K = input_tensor.shape
    N, K_w = weight.shape
    assert K == K_w, f"Dimension mismatch: input K={K}, weight K={K_w}"
    
    # Allocate output
    output = torch.empty((M, N), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    fused_linear_add_kernel[grid](
        input_tensor, weight, bias, residual, output,
        M, N, K,
        input_tensor.stride(0), input_tensor.stride(1),
        weight.stride(0), weight.stride(1),
        residual.stride(0), residual.stride(1),
        output.stride(0), output.stride(1),
    )
    
    # Reshape output to match original input shape
    if len(original_shape) > 2:
        output = output.reshape(*original_shape[:-1], N)
    
    return output


def replacement_func():
    return fused_linear_add