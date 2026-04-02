import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = 0.0625 * in_0
    return tmp_0

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def simple_scale_kernel(x_ptr, output_ptr, stride_0, stride_1, stride_2, m, n, k, scale_factor: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offset < m
    n_mask = n_offset < n
    
    x_ptrs = x_ptr + m_offset[:, None] * stride_0 + n_offset[None, :] * stride_1
    x = tl.load(x_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    
    out = x * scale_factor
    
    output_ptrs = output_ptr + m_offset[:, None] * stride_0 + n_offset[None, :] * stride_1
    tl.store(output_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def simple_scale(x, scale_factor=0.0625):
    if x.dim() != 3:
        raise ValueError("Input must be 3D tensor")
    
    m, n, k = x.shape
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    grid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    output = torch.empty_like(x)
    
    simple_scale_kernel[(grid_m, grid_n)](
        x_ptr=x,
        output_ptr=output,
        stride_0=x.stride(0),
        stride_1=x.stride(1),
        stride_2=x.stride(2),
        m=m,
        n=n,
        k=k,
        scale_factor=scale_factor,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return simple_scale