import torch
import triton
import triton.language as tl

# Pattern matching function - matches conv2d(1x1) + mul(1.0) + reshape
def pattern(bias, weight, input_tensor):
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    mul_out = conv_out * 1.0
    reshape_out = mul_out.reshape(-1, 17, 4096)
    return reshape_out

def replacement_args(bias, weight, input_tensor):
    return (bias, weight, input_tensor)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
    ],
    key=['B', 'M', 'K', 'N'],
)
@triton.jit
def conv1x1_kernel(
    input_ptr,      # [B, K, N]
    weight_ptr,     # [M, K]
    bias_ptr,       # [M]
    output_ptr,     # [B, M, N]
    B, M, K, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    # Initialize accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Accumulate over K using tl.dot
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        
        # Load weight [BLOCK_M, BLOCK_K]
        weight_offset = m_offsets[:, None] * K + k_offsets[None, :]
        weight_mask = m_mask[:, None] & k_mask[None, :]
        weight_vals = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0)
        
        # Load input [BLOCK_K, BLOCK_N]
        input_offset = pid_b * K * N + k_offsets[:, None] * N + n_offsets[None, :]
        input_mask = k_mask[:, None] & n_mask[None, :]
        input_vals = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)
        
        # acc += weight @ input
        acc += tl.dot(weight_vals, input_vals)
    
    # Add bias [BLOCK_M, 1]
    bias_vals = tl.load(bias_ptr + m_offsets, mask=m_mask, other=0.0)
    acc += bias_vals[:, None]
    
    # Store output [BLOCK_M, BLOCK_N]
    output_offset = pid_b * M * N + m_offsets[:, None] * N + n_offsets[None, :]
    output_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + output_offset, acc, mask=output_mask)


# Fixed-config kernel for B=1 - no autotuning overhead
@triton.jit
def conv1x1_kernel_b1(
    input_ptr,      # [K, N]
    weight_ptr,     # [M, K]
    bias_ptr,       # [M]
    output_ptr,     # [M, N]
    M, K, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    # Initialize accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Accumulate over K using tl.dot
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        
        # Load weight [BLOCK_M, BLOCK_K]
        weight_offset = m_offsets[:, None] * K + k_offsets[None, :]
        weight_mask = m_mask[:, None] & k_mask[None, :]
        weight_vals = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0)
        
        # Load input [BLOCK_K, BLOCK_N]
        input_offset = k_offsets[:, None] * N + n_offsets[None, :]
        input_mask = k_mask[:, None] & n_mask[None, :]
        input_vals = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)
        
        # acc += weight @ input
        acc += tl.dot(weight_vals, input_vals)
    
    # Add bias [BLOCK_M, 1]
    bias_vals = tl.load(bias_ptr + m_offsets, mask=m_mask, other=0.0)
    acc += bias_vals[:, None]
    
    # Store output [BLOCK_M, BLOCK_N]
    output_offset = m_offsets[:, None] * N + n_offsets[None, :]
    output_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + output_offset, acc, mask=output_mask)


@torch.fx.wrap
def conv1x1_mul_reshape_fused(bias, weight, input_tensor):
    B = input_tensor.shape[0]
    C_in = input_tensor.shape[1]  # K = 256
    H = input_tensor.shape[2]     # 64
    W = input_tensor.shape[3]     # 64
    C_out = weight.shape[0]       # M = 17
    HW = H * W                    # N = 4096
    
    # Output shape: [B, M, N] = [B, 17, 4096]
    output = torch.empty((B, C_out, HW), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Flatten weight from [17, 256, 1, 1] to [17, 256] - no copy needed since it's contiguous
    weight_flat = weight.view(C_out, C_in)
    
    # Flatten input from [B, 256, 64, 64] to [B, 256, 4096] - no copy needed
    input_flat = input_tensor.view(B, C_in, HW)
    
    if B == 1:
        # Use fixed-config kernel for B=1 - no autotuning overhead
        # BLOCK_M=16 gives better utilization for M=17 (2 blocks instead of 1 with padding)
        BLOCK_M = 16
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (triton.cdiv(C_out, BLOCK_M), triton.cdiv(HW, BLOCK_N))
        conv1x1_kernel_b1[grid](
            input_flat.squeeze(0),  # [K, N]
            weight_flat,
            bias,
            output.squeeze(0),      # [M, N]
            C_out,
            C_in,
            HW,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_warps=4,
            num_stages=3,
        )
    else:
        grid = lambda meta: (B, triton.cdiv(C_out, meta['BLOCK_M']), triton.cdiv(HW, meta['BLOCK_N']))
        conv1x1_kernel[grid](
            input_flat,
            weight_flat,
            bias,
            output,
            B,
            C_out,
            C_in,
            HW,
        )
    
    return output


def replacement_func():
    return conv1x1_mul_reshape_fused