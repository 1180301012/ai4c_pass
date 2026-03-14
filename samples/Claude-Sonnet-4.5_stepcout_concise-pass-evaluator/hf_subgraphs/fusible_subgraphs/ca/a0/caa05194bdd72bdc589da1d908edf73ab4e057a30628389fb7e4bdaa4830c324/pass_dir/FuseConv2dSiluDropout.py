import torch
import triton
import triton.language as tl

def pattern(bias, weight, input_tensor):
    """
    Pattern matching Conv2d with 1x1 kernel
    """
    output = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    return output

def replacement_args(bias, weight, input_tensor):
    return (bias, weight, input_tensor)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    1x1 Convolution kernel treating it as matrix multiplication
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

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = input_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = weight_ptr + (offs_bn[:, None] * K + offs_k[None, :])
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        acc += tl.dot(a, b.T)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    
    if bias_ptr is not None:
        offs_cn = offs_bn
        c = acc + tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)[None, :]
    else:
        c = acc
    
    offs_cm = offs_am[:, None]
    offs_cn = offs_bn[None, :]
    c_ptrs = output_ptr + offs_cm * N + offs_cn
    c_mask = (offs_cm < M) & (offs_cn < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def conv1x1_nchw_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C_in, H, W, C_out,
    BLOCK_SIZE: tl.constexpr,
):
    """Direct 1x1 conv on NCHW layout"""
    # Each program handles BLOCK_SIZE output elements
    pid = tl.program_id(0)
    num_elements = B * C_out * H * W
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Decode output position: n, c_out, h, w
    n = offsets // (C_out * H * W)
    remainder = offsets % (C_out * H * W)
    c_out = remainder // (H * W)
    remainder = remainder % (H * W)
    h = remainder // W
    w = remainder % W
    
    # Compute dot product over C_in
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for c_in in range(C_in):
        input_idx = n * (C_in * H * W) + c_in * (H * W) + h * W + w
        weight_idx = c_out * C_in + c_in
        
        input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        weight_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
        acc += input_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + c_out, mask=mask, other=0.0)
    result = acc + bias_val
    
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_conv1x1(bias, weight, input_tensor):
    """Optimized 1x1 convolution working directly on NCHW"""
    B, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    
    output = torch.empty((B, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    num_elements = B * C_out * H * W
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    conv1x1_nchw_kernel[grid](
        input_tensor, weight, bias, output,
        B, C_in, H, W, C_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_conv1x1