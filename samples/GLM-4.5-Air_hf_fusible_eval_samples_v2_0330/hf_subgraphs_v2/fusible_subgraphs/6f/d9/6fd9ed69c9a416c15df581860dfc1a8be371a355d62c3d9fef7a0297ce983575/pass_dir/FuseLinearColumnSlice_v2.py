import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Pattern: Linear operation followed by column slicing for first and last 256 columns"""
    # First linear operation
    tmp_4 = torch.nn.functional.linear(in_5, in_1, in_0)
    
    # Column slicing operations
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 256, None)]  # First 256 columns
    tmp_6 = tmp_5.view(-1, 256)
    tmp_7 = tmp_4[slice(None, None, None), slice(-256, None, None)]  # Last 256 columns  
    tmp_8 = tmp_7.view(-1, 256)
    
    # Second linear operation
    tmp_9 = in_4.reshape(300, -1, 256)
    tmp_10 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    
    # Final slicing and operations
    tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]  # First 256 channels
    tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]  # Last 256 channels
    tmp_13 = tmp_6.unsqueeze(-2)
    
    return (tmp_11, tmp_12, tmp_8, tmp_13)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

@triton.jit
def simple_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Simple Triton add kernel for testing"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_triton_add(x, y):
    """Simple Triton add function"""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    simple_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

@triton.jit
def linear_kernel(
    x_ptr, x_stride,
    weight_ptr, weight_row_stride, weight_col_stride,
    bias_ptr,
    out_ptr, out_stride,
    N, K, M,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """Triton kernel for linear operation (Y = X @ W^T + b)"""
    # Program identifiers
    pid = tl.program_id(0)
    M_offset = pid * BLOCK_M
    
    # Bounds
    M_mask = (M_offset + tl.arange(0, BLOCK_M)) < N
    
    # Process with block-based computation
    for k_offset in range(0, K, BLOCK_K):
        k_bound = min(k_offset + BLOCK_K, K)
        k_mask = tl.arange(k_offset, k_bound)
        
        # Load input block
        x = tl.load(x_ptr + M_offset * x_stride + 
                   k_mask[None, :] * x_stride,
                   mask=M_mask[:, None] & (k_mask[None, :] < K))
        
        # Load weight block (transpose W for efficient computation)
        weight = tl.load(weight_ptr + 
                        tl.arange(0, M)[:, None] * weight_row_stride + 
                        k_mask[None, :] * weight_col_stride,
                mask=(tl.arange(0, M)[:, None] < M) & (k_mask[None, :] < K))
        
        # Load bias
        bias = tl.load(bias_ptr + tl.arange(0, M), mask=(tl.arange(0, M) < M))
        
        # Compute matrix multiplication
        acc = tl.dot(x.to(tl.float32), weight.to(tl.float32)).to(tl.float32) + bias[None, :]
        
        # Store result
        tl.store(out_ptr + (M_offset + tl.arange(0, BLOCK_M))[:, None] * out_stride + 
                tl.arange(0, M)[None, :], acc, mask=M_mask[:, None] & (tl.arange(0, M)[None, :] < M))

@torch.fx.wrap
def optimized_linear(x, weight, bias):
    """Optimized linear operation using Triton kernel"""
    N, K = x.shape
    M = weight.shape[0]
    
    # Create output tensor
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 32
    grid_size = (triton.cdiv(N, BLOCK_M),)
    
    linear_kernel[grid_size](
        x, x.stride(0),
        weight, weight.stride(0), weight.stride(1),
        bias,
        out, out.stride(1),
        N, K, M,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return out

@torch.fx.wrap
def optimized_fusion_linear(in_0, in_1, in_2, in_3, in_4, in_5):
    """Optimized version with Triton kernels"""
    # First linear operation
    tmp_4 = optimized_linear(in_5, in_1, in_0)
    
    # Column slicing operations
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 256, None)]
    tmp_6 = tmp_5.view(-1, 256)
    tmp_7 = tmp_4[slice(None, None, None), slice(-256, None, None)]
    tmp_8 = tmp_7.view(-1, 256)
    
    # Second linear operation
    tmp_9 = in_4.reshape(300, -1, 256)
    tmp_10 = optimized_linear(tmp_9, in_3, in_2)
    tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]
    tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]
    tmp_13 = tmp_6.unsqueeze(-2)
    
    return (tmp_11, tmp_12, tmp_8, tmp_13)

def replacement_func():
    return optimized_fusion_linear