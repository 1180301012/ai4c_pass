import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_kernel(
    bias_ptr, weight_ptr, input_ptr, output_ptr,
    M, N, K,
    stride_bias,
    stride_weight_row, stride_weight_col,
    stride_input_row, stride_input_col,
    stride_output_row, stride_output_col,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute: output = input @ weight^T + bias
    # input: [M, K], weight: [N, K], bias: [N], output: [M, N]
    
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Masks for boundary handling
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < K
        
        # Load input block: [BLOCK_M, BLOCK_K]
        input_ptrs = input_ptr + offs_m[:, None] * stride_input_row + k_offs[None, :] * stride_input_col
        input_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Load weight^T block: [BLOCK_K, BLOCK_N]
        # weight[n, k] with axes swapped to get weight^T[k, n]
        weight_T_ptrs = weight_ptr + k_offs[:, None] * stride_weight_col + offs_n[None, :] * stride_weight_row
        weight_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(weight_T_ptrs, mask=weight_mask, other=0.0)
        
        # Compute dot product: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] = [BLOCK_M, BLOCK_N]
        accumulator += tl.dot(a, b)
    
    # Add bias
    bias_ptrs = bias_ptr + offs_n * stride_bias
    bias = tl.load(bias_ptrs, mask=mask_n, other=0.0).to(tl.float32)
    accumulator += bias[None, :]
    
    # Store output
    output_ptrs = output_ptr + offs_m[:, None] * stride_output_row + offs_n[None, :] * stride_output_col
    output_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(output_ptrs, accumulator, mask=output_mask)


@torch.fx.wrap
def fused_linear_dispatch(bias, weight, input_tensor, route):
    # Compute: output = input_tensor @ weight^T + bias
    # Handles batched inputs without reshape
    
    batch_dims = input_tensor.shape[:-1]
    M = 1
    for d in batch_dims:
        M = M * d
    K = input_tensor.shape[-1]
    N = weight.shape[0]
    
    output_shape = list(batch_dims) + [N]
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    stride_input_row = input_tensor.stride(-2)
    stride_input_col = input_tensor.stride(-1)
    stride_output_row = output.stride(-2)
    stride_output_col = output.stride(-1)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    fused_linear_kernel[grid](
        bias_ptr=bias, weight_ptr=weight, input_ptr=input_tensor, output_ptr=output,
        M=M, N=N, K=K,
        stride_bias=1 if bias.ndim == 1 else bias.stride(0),
        stride_weight_row=weight.stride(0), stride_weight_col=weight.stride(1),
        stride_input_row=stride_input_row, stride_input_col=stride_input_col,
        stride_output_row=stride_output_row, stride_output_col=stride_output_col,
    )
    
    return output


def get_replacement_func():
    return fused_linear_dispatch