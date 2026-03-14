import torch
import triton
import triton.language as tl

# Pattern: conv2d with 1x1 kernel, stride (2,2)
def pattern(input, weight):
    conv_out = torch.conv2d(input, weight, None, (2, 2), (0, 0), (1, 1), 1)
    return conv_out

def replacement_args(input, weight):
    return (input, weight)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_stride2_kernel(
    input_ptr, weight_ptr, output_ptr,
    N_batch, C_in, C_out, H_in, W_in, H_out, W_out,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_w_o, stride_w_i,
    stride_out_n, stride_out_c, stride_out_hw,
    M, N, K,  # M = N_batch * H_out * W_out, N = C_out, K = C_in
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute 1x1 convolution with stride 2.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    HW_out = H_out * W_out
    
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        
        # Compute batch and spatial indices
        batch_idx = offs_m // HW_out
        hw_idx = offs_m % HW_out
        h_out = hw_idx // W_out
        w_out = hw_idx % W_out
        
        # Map to input coordinates (stride 2)
        h_in = h_out * 2
        w_in = w_out * 2
        
        # Load input tile
        input_ptrs = input_ptr + batch_idx[:, None] * stride_in_n + k_offs[None, :] * stride_in_c + h_in[:, None] * stride_in_h + w_in[:, None] * stride_in_w
        mask_input = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        input_tile = tl.load(input_ptrs, mask=mask_input, other=0.0)
        
        # Load weight tile
        weight_ptrs = weight_ptr + offs_n[None, :] * stride_w_o + k_offs[:, None] * stride_w_i
        mask_weight = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        weight_tile = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(input_tile, weight_tile)
    
    # Store output
    batch_idx = offs_m // HW_out
    hw_idx = offs_m % HW_out
    output_ptrs = output_ptr + batch_idx[:, None] * stride_out_n + offs_n[None, :] * stride_out_c + hw_idx[:, None] * stride_out_hw
    mask_output = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, acc, mask=mask_output)


@torch.fx.wrap
def conv1x1_stride2_triton(input, weight):
    """
    Optimized 1x1 convolution with stride 2 using Triton.
    """
    N_batch, C_in, H_in, W_in = input.shape
    C_out = weight.shape[0]
    
    # Output spatial dimensions with stride 2
    H_out = (H_in + 1) // 2
    W_out = (W_in + 1) // 2
    HW_out = H_out * W_out
    
    # Ensure weight is on the same device as input
    if weight.device != input.device:
        weight = weight.to(input.device)
    
    input = input.contiguous()
    weight_2d = weight.view(C_out, C_in).contiguous()
    
    output = torch.empty((N_batch, C_out, H_out, W_out), device=input.device, dtype=input.dtype)
    
    M = N_batch * HW_out
    N = C_out
    K = C_in
    
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    conv1x1_stride2_kernel[grid](
        input, weight_2d, output,
        N_batch, C_in, C_out, H_in, W_in, H_out, W_out,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight_2d.stride(0), weight_2d.stride(1),
        C_out * HW_out, HW_out, 1,
        M, N, K,
    )
    
    return output


def replacement_func():
    return conv1x1_stride2_triton