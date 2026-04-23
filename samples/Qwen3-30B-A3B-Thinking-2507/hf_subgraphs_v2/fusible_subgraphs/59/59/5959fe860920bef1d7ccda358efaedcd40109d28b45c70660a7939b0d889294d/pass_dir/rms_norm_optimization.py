import torch
import triton
import triton.language as tl

def pattern(x, weight):
    x_cast = x.to(torch.float32)
    x_sq = x_cast.pow(2)
    x_mean = x_sq.mean(-1, keepdim=True)
    x_eps = x_mean + 1e-06
    x_rsqrt = torch.rsqrt(x_eps)
    x_scaled = x_cast * x_rsqrt
    return x_scaled

def replacement_args(x, weight):
    return (x, weight)

@triton.jit
@torch.fx.wrap
def rms_norm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    B, M, N,
    BLOCK_SIZE: tl.constexpr,
):
    b_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    
    # Calculate base index for this (b, m) slice
    base_idx = b_idx * M * N + m_idx * N
    
    # Block for the entire N dimension
    start = 0
    end = N
    
    # Calculate mean across N
    sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(start, end):
        # Calculate index in the first N elements
        idx = base_idx + i
        x_val = tl.load(x_ptr + idx)
        sum_sq[i - start] = x_val * x_val
    
    # Compute mean within block
    sum_sq = tl.sum(sum_sq, axis=0) / N
    
    # Calculate denominator for normalization
    denom = tl.rsqrt(sum_sq + 1e-6)
    
    # Scale and store
    for i in range(start, end):
        idx = base_idx + i
        x_val = tl.load(x_ptr + idx)
        out_val = x_val * denom
        tl.store(out_ptr + idx, out_val)

@torch.fx.wrap
def rms_norm(x, weight):
    B, M, N = x.shape
    out = torch.empty((B, M, N), dtype=torch.bfloat16, device=x.device)
    
    grid = (B, M)
    rms_norm_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        out_ptr=out,
        B=B,
        M=M,
        N=N,
        BLOCK_SIZE=128,
    )
    return out

def replacement_func():
    return rms_norm