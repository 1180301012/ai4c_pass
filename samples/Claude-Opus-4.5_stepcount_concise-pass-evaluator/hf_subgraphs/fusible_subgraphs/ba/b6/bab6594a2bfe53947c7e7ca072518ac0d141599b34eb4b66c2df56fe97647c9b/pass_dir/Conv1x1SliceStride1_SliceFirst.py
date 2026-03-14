import torch
import triton
import triton.language as tl

# Pattern: conv2d with 1x1 kernel, stride (1,1)
def pattern(input, weight):
    conv_out = torch.conv2d(input, weight, None, (1, 1), (0, 0), (1, 1), 1)
    return conv_out

def replacement_args(input, weight):
    return (input, weight)


@triton.autotune(
    configs=[
        # Smaller blocks for small batch sizes
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        # Medium blocks
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        # Larger blocks for large batch sizes  
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    input_ptr, weight_ptr, output_ptr,
    N_batch, C_in, C_out, HW,
    stride_in_n, stride_in_c, stride_in_hw,
    stride_w_o, stride_w_i,
    stride_out_n, stride_out_c, stride_out_hw,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        
        batch_idx = offs_m // HW
        hw_idx = offs_m % HW
        
        input_ptrs = input_ptr + batch_idx[:, None] * stride_in_n + k_offs[None, :] * stride_in_c + hw_idx[:, None] * stride_in_hw
        mask_input = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        input_tile = tl.load(input_ptrs, mask=mask_input, other=0.0)
        
        weight_ptrs = weight_ptr + offs_n[None, :] * stride_w_o + k_offs[:, None] * stride_w_i
        mask_weight = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        weight_tile = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(input_tile, weight_tile)
    
    batch_idx = offs_m // HW
    hw_idx = offs_m % HW
    output_ptrs = output_ptr + batch_idx[:, None] * stride_out_n + offs_n[None, :] * stride_out_c + hw_idx[:, None] * stride_out_hw
    mask_output = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, acc, mask=mask_output)


@torch.fx.wrap
def conv1x1_triton(input, weight):
    N_batch, C_in, H, W = input.shape
    C_out = weight.shape[0]
    HW = H * W
    
    if weight.device != input.device:
        weight = weight.to(input.device)
    
    input = input.contiguous()
    weight_2d = weight.view(C_out, C_in).contiguous()
    
    output = torch.empty((N_batch, C_out, H, W), device=input.device, dtype=input.dtype)
    
    M = N_batch * HW
    N = C_out
    K = C_in
    
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    conv1x1_kernel[grid](
        input, weight_2d, output,
        N_batch, C_in, C_out, HW,
        C_in * HW, HW, 1,
        weight_2d.stride(0), weight_2d.stride(1),
        C_out * HW, HW, 1,
        M, N, K,
    )
    
    return output


def replacement_func():
    return conv1x1_triton