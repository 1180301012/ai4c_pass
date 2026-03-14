import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: conv2d -> stack -> sum -> cat
    The stack+sum is a no-op that can be eliminated
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.stack([tmp_2], dim=0)
    tmp_4 = tmp_3.sum(dim=0)
    tmp_5 = torch.cat([tmp_4, in_2], 1)
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused matrix multiplication + bias: C = A @ B + bias
    A: [M, K], B: [K, N], bias: [M], C: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        mask_b = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        acc += tl.dot(a, b)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Add bias
    bias_mask = offs_m < M
    bias = tl.load(bias_ptr + offs_m, mask=bias_mask, other=0.0)
    acc += bias[:, None]
    
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def add_bias_kernel(
    input_ptr, bias_ptr, output_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Add bias and copy to output: output[m, n] = input[m, n] + bias[m]
    """
    pid = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    n_start = pid * BLOCK_SIZE
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
    n_mask = n_offsets < N
    
    # Load bias
    bias_val = tl.load(bias_ptr + pid_m)
    
    # Load input, add bias
    input_offsets = pid_m * N + n_offsets
    input_vals = tl.load(input_ptr + input_offsets, mask=n_mask, other=0.0)
    output_vals = input_vals + bias_val
    
    # Store to output
    tl.store(output_ptr + input_offsets, output_vals, mask=n_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def copy_kernel(
    src_ptr, dst_ptr,
    M, N, dst_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Copy tensor with channel offset for concatenation
    """
    pid = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    n_start = pid * BLOCK_SIZE
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
    n_mask = n_offsets < N
    
    # Load from source
    src_offsets = pid_m * N + n_offsets
    vals = tl.load(src_ptr + src_offsets, mask=n_mask, other=0.0)
    
    # Store to destination with offset
    dst_offsets = (pid_m + dst_offset) * N + n_offsets
    tl.store(dst_ptr + dst_offsets, vals, mask=n_mask)

@torch.fx.wrap
def fused_conv_cat(bias, weight, other_tensor, input_tensor):
    """
    Fused 1x1 conv2d + cat using optimized Triton kernels
    
    Args:
        bias: [C_out]
        weight: [C_out, C_in, 1, 1]
        other_tensor: [B, C_out, H, W]
        input_tensor: [B, C_in, H, W]
    
    Returns:
        output: [B, 2*C_out, H, W]
    """
    B, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    N = H * W
    
    # Reshape for matrix multiplication
    # input: [B, C_in, H, W] -> [B, C_in, H*W]
    # weight: [C_out, C_in, 1, 1] -> [C_out, C_in]
    input_2d = input_tensor.reshape(B, C_in, N).contiguous()
    weight_2d = weight.reshape(C_out, C_in).contiguous()
    
    # Allocate output directly
    output = torch.empty((B, 2 * C_out, H, W), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Get view of first half of output for conv result
    output_conv_2d = output[:, :C_out, :, :].reshape(B, C_out, N)
    
    # Step 1: Fused matrix multiplication + bias - write directly to output
    for b in range(B):
        grid = (triton.cdiv(C_out, 128), triton.cdiv(N, 128))
        matmul_bias_kernel[grid](
            weight_2d, input_2d[b], bias, output_conv_2d[b],
            C_out, N, C_in,
            C_in, 1,  # weight strides: [C_out, C_in]
            N, 1,     # input strides: [C_in, N]
            N, 1,     # output strides: [C_out, N]
        )
    
    # Step 2: Copy other_tensor to second half
    output[:, C_out:, :, :] = other_tensor
    
    return output

def replacement_func():
    return fused_conv_cat