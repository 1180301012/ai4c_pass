import torch
import triton
import triton.language as tl

def pattern(x, w, b):
    """Pattern: Linear operation that could benefit from optimization"""
    return torch.nn.functional.linear(x, w, b)

def replacement_args(x, w, b):
    return (x, w, b)

@triton.jit
def linear_kernel(
    x_ptr, x_stride,
    weight_ptr, weight_row_stride, weight_col_stride,
    bias_ptr,
    out_ptr, out_stride,
    N, K, M,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    """Simplified Triton kernel for linear operation"""
    pid = tl.program_id(0)
    M_offset = pid * BLOCK_M
    M_mask = (M_offset + tl.arange(0, BLOCK_M)) < N
    
    # Process all K dimension in one go for simplicity
    k_indices = tl.arange(0, BLOCK_K)
    k_mask = k_indices < K
    
    # Load input block
    x = tl.load(x_ptr + M_offset * x_stride + 
               k_indices[None, :] * x_stride,
               mask=M_mask[:, None] & k_mask[None, :])
    
    # Load weights for this block
    weight = tl.load(weight_ptr + 
                    tl.arange(0, M)[:, None] * weight_row_stride + 
                    k_indices[None, :] * weight_col_stride,
            mask=(tl.arange(0, M)[:, None] < M) & k_mask[None, :])
    
    # Load bias
    bias = tl.load(bias_ptr + tl.arange(0, M), mask=(tl.arange(0, M) < M))
    
    # Compute matrix multiplication
    acc = tl.dot(x.to(tl.float32), weight.to(tl.float32)).to(tl.float32) + bias[None, :]
    
    # Store result
    tl.store(out_ptr + (M_offset + tl.arange(0, BLOCK_M))[:, None] * out_stride + 
            tl.arange(0, M)[None, :], acc, mask=M_mask[:, None] & (tl.arange(0, M)[None, :] < M))

@torch.fx.wrap
def optimized_linear(x, w, b):
    """Linear operation optimized with Triton"""
    N, K = x.shape
    M = w.shape[0]
    
    # Create output tensor
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    
    # Launch kernel with optimized block sizes
    BLOCK_M = 64
    BLOCK_K = min(256, K)  # Make sure BLOCK_K doesn't exceed K
    grid_size = (triton.cdiv(N, BLOCK_M),)
    
    linear_kernel[grid_size](
        x, x.stride(0),
        w, w.stride(0), w.stride(1),
        b,
        out, out.stride(1),
        N, K, M,
        BLOCK_M, BLOCK_K
    )
    
    return out

def replacement_func():
    return optimized_linear