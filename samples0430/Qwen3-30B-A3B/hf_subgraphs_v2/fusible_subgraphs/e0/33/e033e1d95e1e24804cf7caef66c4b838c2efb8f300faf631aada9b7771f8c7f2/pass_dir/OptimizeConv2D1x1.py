import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_3, in_1, in_0):
    result = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return result

# Argument extraction function

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

# Optimized convolution kernel
@triton.jit
def conv2d_1x1_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    B, H, W, C_in,
    C_out: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_m = pid_m * BLOCK_M
    block_n = pid_n * BLOCK_N
    
    m_offsets = block_m + tl.arange(0, BLOCK_M)
    n_offsets = block_n + tl.arange(0, BLOCK_N)
    
    valid_m = m_offsets < B * H * W
    valid_n = n_offsets < C_out
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, C_in, BLOCK_K):
        input_ptr = x_ptr + (m_offsets[:, None] * C_in + k)
        input = tl.load(
            input_ptr,
            mask=valid_m[:, None] & (tl.arange(0, BLOCK_K) < C_in - k),
            other=0.0
        )
        
        weight_ptr = w_ptr + k + tl.arange(0, BLOCK_K)[:, None] * C_in + tl.arange(0, BLOCK_N)[None, :]
        weight = tl.load(
            weight_ptr,
            mask=(tl.arange(0, BLOCK_K)[:, None] < C_in - k) & valid_n[None, :],
            other=0.0
        )
        
        accumulator += tl.dot(input, weight)

    bias = tl.load(bias_ptr + tl.arange(0, C_out), mask=tl.full((C_out,), True, dtype=tl.int1))
    bias_values = tl.load(bias_ptr + (block_n * BLOCK_N) + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < C_out - block_n * BLOCK_N, other=0.0)
    accumulator = accumulator + bias_values[None, :]
    
    out_ptr = out_ptr + (m_offsets[:, None] * C_out + n_offsets)
    tl.store(out_ptr, accumulator, mask=valid_m[:, None] & valid_n)


@torch.fx.wrap
def kernel_wrapper(x, w, b):
    B, C_in, H, W = x.shape
    C_out = w.shape[0]
    out = torch.empty((B, C_out, H, W), dtype=x.dtype, device=x.device)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    grid_m = triton.cdiv(B * H * W, BLOCK_M)
    grid_n = triton.cdiv(C_out, BLOCK_N)
    
    conv2d_1x1_kernel[(grid_m, grid_n)](
        x,
        w,
        b,
        out,
        B, H, W, C_in, C_out,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    return out


def replacement_func():
    return kernel_wrapper