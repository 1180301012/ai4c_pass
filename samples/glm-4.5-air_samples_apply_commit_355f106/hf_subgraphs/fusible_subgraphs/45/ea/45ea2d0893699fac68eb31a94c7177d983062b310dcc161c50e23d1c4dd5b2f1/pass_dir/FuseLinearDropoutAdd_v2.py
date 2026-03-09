import torch
import triton
import triton.language as tl


def pattern(bias, weight, residual, input_tensor):
    linear_out = torch.nn.functional.linear(input_tensor, weight, bias)
    dropout_out = torch.nn.functional.dropout(linear_out, p=0.0, training=False)
    add_out = residual + dropout_out
    return add_out


def replacement_args(bias, weight, residual, input_tensor):
    return (bias, weight, residual, input_tensor)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_kernel(input_ptr, weight_ptr, bias_ptr, residual_ptr, output_ptr, dropout_out_ptr,
                 M, N, K, stride_im, stride_ik, stride_wk, stride_wn, stride_om, stride_on,
                 stride_rm, stride_rn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    weight_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(input_ptrs, mask=mask, other=0.0)
        b = tl.load(weight_ptrs, mask=mask, other=0.0)
        accumulator += tl.dot(a, b)
        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk
        offs_k += BLOCK_SIZE_K
    
    bias = tl.load(bias_ptr + offs_n)
    accumulator = accumulator + bias
    
    residual_ptrs = residual_ptr + (offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn)
    residual = tl.load(residual_ptrs, mask=True, other=0.0)
    output = accumulator + residual
    
    output_ptrs = output_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(output_ptrs, output, mask=True)
    
    dropout_out = accumulator
    dropout_out_ptrs = dropout_out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(dropout_out_ptrs, dropout_out, mask=True)


@torch.fx.wrap
def fused_wrapper(bias, weight, residual, input_tensor):
    M, K = input_tensor.shape
    K_w, N = weight.shape
    
    output = torch.empty((M, N), device=input_tensor.device, dtype=input_tensor.dtype)
    dropout_out = torch.empty((M, N), device=input_tensor.device, dtype=input_tensor.dtype)
    
    def grid(M, N, K):
        return (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
    
    fused_kernel[grid](
        input_tensor, weight, bias, residual, output, dropout_out,
        M, N, K,
        input_tensor.stride(0), input_tensor.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        residual.stride(0), residual.stride(1),
    )
    
    return output, dropout_out


def replacement_func():
    return fused_wrapper