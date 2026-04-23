import torch
import triton
import triton.language as tl

def pattern(x):
    t = torch.cat((x, x), dim = -1)
    cos_t = t.cos() * 1.0
    sin_t = t.sin() * 1.0
    return cos_t, sin_t

def replacement_args(x):
    return (x,)

@triton.jit
def sin_cos_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    B, M, N,
    BLOCK_SIZE: tl.constexpr,
):
    b_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    block_idx = tl.program_id(2)
    
    # Calculate base index for this (b, m) slice
    base_idx = b_idx * M * 2 * N + m_idx * 2 * N
    
    start = block_idx * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, N)
    
    for i in range(start, end):
        # Compute index in the first N elements
        x_idx = b_idx * M * N + m_idx * N + i
        x_val = tl.load(x_ptr + x_idx)
        cos_val = tl.cos(x_val)
        sin_val = tl.sin(x_val)
        
        # Store in both halves of the concatenated tensor
        idx_cos = base_idx + i
        idx_sin = idx_cos
        tl.store(cos_ptr + idx_cos, cos_val)
        tl.store(sin_ptr + idx_sin, sin_val)
        
        # Store second half (shifted by N)
        tl.store(cos_ptr + idx_cos + N, cos_val)
        tl.store(sin_ptr + idx_sin + N, sin_val)

@torch.fx.wrap
def sin_cos(x):
    B, M, N = x.shape
    cos_out = torch.empty((B, M, 2 * N), dtype=x.dtype, device=x.device)
    sin_out = torch.empty((B, M, 2 * N), dtype=x.dtype, device=x.device)
    
    grid = (B, M)
    num_blocks = (N + 127) // 128
    grid = (B, M, num_blocks)
    sin_cos_kernel[grid](
        x_ptr=x,
        cos_ptr=cos_out,
        sin_ptr=sin_out,
        B=B,
        M=M,
        N=N,
        BLOCK_SIZE=128,
    )
    return cos_out, sin_out

def replacement_func():
    return sin_cos