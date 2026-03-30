import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Simple pattern to match: just linear operation"""
    # First linear operation
    tmp_4 = torch.nn.functional.linear(in_5, in_1, in_0)
    
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

@triton.jit
def simple_linear_kernel(
    x_ptr, x_stride,
    weight_ptr, weight_row_stride, weight_col_stride,
    bias_ptr,
    out_ptr, out_stride,
    N, K, M,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """Simple Triton kernel for linear operation"""
    pid = tl.program_id(0)
    M_offset = pid * BLOCK_M
    M_mask = (M_offset + tl.arange(0, BLOCK_M)) < N
    
    for k_offset in range(0, K, BLOCK_K):
        k_bound = min(k_offset + BLOCK_K, K)
        k_mask = tl.arange(k_offset, k_bound)
        
        # Load input
        x = tl.load(x_ptr + M_offset * x_stride + 
                   k_mask[None, :] * x_stride,
                   mask=M_mask[:, None] & (k_mask[None, :] < K))
        
        # Load weights
        weight = tl.load(weight_ptr + 
                        tl.arange(0, M)[:, None] * weight_row_stride + 
                        k_mask[None, :] * weight_col_stride,
                mask=(tl.arange(0, M)[:, None] < M) & (k_mask[None, :] < K))
        
        # Load bias
        bias = tl.load(bias_ptr + tl.arange(0, M), mask=(tl.arange(0, M) < M))
        
        # Compute
        acc = tl.dot(x.to(tl.float32), weight.to(tl.float32)).to(tl.float32) + bias[None, :]
        
        # Store
        tl.store(out_ptr + (M_offset + tl.arange(0, BLOCK_M))[:, None] * out_stride + 
                tl.arange(0, M)[None, :], acc, mask=M_mask[:, None] & (tl.arange(0, M)[None, :] < M))

@torch.fx.wrap
def optimized_linear_simple(x, weight, bias):
    """Simple optimized linear operation"""
    N, K = x.shape
    M = weight.shape[0]
    
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 32
    grid_size = (triton.cdiv(N, BLOCK_M),)
    
    simple_linear_kernel[grid_size](
        x, x.stride(0),
        weight, weight.stride(0), weight.stride(1),
        bias,
        out, out.stride(1),
        N, K, M,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return out

@torch.fx.wrap
def debug_simple_replacement(in_0, in_1, in_2, in_3, in_4, in_5):
    """Simple replacement function"""
    tmp_4 = optimized_linear_simple(in_5, in_1, in_0)
    return tmp_4

def replacement_func():
    return debug_simple_replacement