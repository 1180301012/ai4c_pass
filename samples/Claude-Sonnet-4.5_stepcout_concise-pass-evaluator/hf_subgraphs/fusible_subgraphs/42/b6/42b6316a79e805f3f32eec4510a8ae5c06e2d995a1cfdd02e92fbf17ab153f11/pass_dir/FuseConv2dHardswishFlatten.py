import torch
import triton
import triton.language as tl


def pattern(bias, weight, input_tensor):
    """
    Pattern to match: Conv2d (1x1) + Hardswish + Flatten
    """
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    hardswish_out = torch.nn.functional.hardswish(conv_out, True)
    flatten_out = hardswish_out.flatten(1, -1)
    return flatten_out


def replacement_args(bias, weight, input_tensor):
    return (bias, weight, input_tensor)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_hardswish_kernel(
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
    Fused matmul + bias + hardswish kernel
    Compute: output[m, n] = hardswish(input[m, k] @ weight[n, k].T + bias[n])
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
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    weight_ptrs = weight_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        mask_k = offs_k < K - k
        mask_input = (offs_m[:, None] < M) & mask_k[None, :]
        mask_weight = (offs_n[:, None] < N) & mask_k[None, :]
        
        input_vals = tl.load(input_ptrs, mask=mask_input, other=0.0)
        weight_vals = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        accumulator += tl.dot(input_vals, weight_vals.T)
        
        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk
    
    # Add bias
    bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    result = accumulator + bias_vals[None, :]
    
    # Apply hardswish: x * clip(x + 3, 0, 6) / 6
    x_plus_3 = result + 3.0
    clipped = tl.maximum(0.0, tl.minimum(6.0, x_plus_3))
    hardswish_result = result * clipped / 6.0
    
    # Store result
    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + offs_om[:, None] * stride_om + offs_on[None, :] * stride_on
    mask_out = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(output_ptrs, hardswish_result, mask=mask_out)


@torch.fx.wrap
def fused_conv1x1_hardswish_flatten(bias, weight, input_tensor):
    """
    Wrapper for fused Conv2d (1x1) + Hardswish + Flatten
    """
    # Get dimensions
    batch_size, in_channels, h, w = input_tensor.shape
    out_channels, _, _, _ = weight.shape
    
    # Reshape: 1x1 conv is essentially matmul
    # input: [B, C_in, 1, 1] -> [B, C_in]
    # weight: [C_out, C_in, 1, 1] -> [C_out, C_in]
    # output = input @ weight.T + bias -> [B, C_out]
    input_2d = input_tensor.squeeze(-1).squeeze(-1).contiguous()
    weight_2d = weight.squeeze(-1).squeeze(-1).contiguous()
    
    M, K = input_2d.shape
    N, _ = weight_2d.shape
    
    output = torch.empty((M, N), device=input_tensor.device, dtype=input_tensor.dtype)
    
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    
    matmul_hardswish_kernel[grid](
        input_2d, weight_2d, bias, output,
        M, N, K,
        input_2d.stride(0), input_2d.stride(1),
        weight_2d.stride(0), weight_2d.stride(1),
        output.stride(0), output.stride(1),
    )
    
    return output


def replacement_func():
    return fused_conv1x1_hardswish_flatten