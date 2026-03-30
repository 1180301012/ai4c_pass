import torch
import triton
import triton.language as tl

def tmp_4, tmp_5, tmp_6, tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, tmp_12, tmp_13):
    """Pattern: Linear operation followed by column slicing for first and last 256 columns"""
    # Create intermediate variables to match the exact pattern
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    
    # First linear operation
    tmp_4 = torch.nn.functional.linear(in_5, tmp_1, tmp_0)
    
    # Column slicing operations
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 256, None)]
    tmp_6 = tmp_5.view(-1, 256)
    tmp_7 = tmp_4[slice(None, None, None), slice(-256, None, None)]
    tmp_8 = tmp_7.view(-1, 256)
    
    # Second linear operation
    tmp_9 = in_4.reshape(300, -1, 256)
    tmp_10 = torch.nn.functional.linear(tmp_9, tmp_3, tmp_2)
    
    # Final operations
    tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]
    tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]
    tmp_13 = tmp_6.unsqueeze(-2)
    
    return (tmp_11, tmp_12, tmp_8, tmp_13)

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
def simple_linear_kernel(
    x_ptr, x_stride,
    weight_first_ptr, weight_first_row_stride, weight_first_col_stride,
    weight_last_ptr, weight_last_row_stride, weight_last_col_stride,
    bias_first_ptr, bias_last_ptr,
    out_first_ptr, out_first_stride,
    out_last_ptr, out_last_stride,
    N, K, M_first, M_last,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    """Simple kernel computing only first and last M columns of linear operation"""
    # Program identifiers
    pid = tl.program_id(0)
    M_offset = pid * BLOCK_M
    
    # Bounds
    M_mask = (M_offset + tl.arange(0, BLOCK_M)) < N
    
    # Process with block-based computation
    for k_offset in range(0, K, BLOCK_K):
        # Compute valid K range for this block
        k_bound = min(k_offset + BLOCK_K, K)
        k_mask = tl.arange(k_offset, k_bound)
        
        # Load input data
        x = tl.load(x_ptr + M_offset * x_stride + 
                   k_mask[None, :] * x_stride,
                   mask=M_mask[:, None] & (k_mask[None, :] < K))
        
        # Load first M columns of weights
        weight_first = tl.load(weight_first_ptr + 
                              tl.arange(0, M_first)[:, None] * weight_first_row_stride + 
                              k_mask[None, :] * weight_first_col_stride,
                mask=(tl.arange(0, M_first)[:, None] < M_first) & (k_mask[None, :] < K))
        
        # Load last M columns of weights
        weight_last = tl.load(weight_last_ptr + 
                             tl.arange(0, M_last)[:, None] * weight_last_row_stride + 
                             k_mask[None, :] * weight_last_col_stride,
                mask=(tl.arange(0, M_last)[:, None] < M_last) & (k_mask[None, :] < K))
        
        # Load biases
        bias_first = tl.load(bias_first_ptr + tl.arange(0, M_first), mask=(tl.arange(0, M_first) < M_first))
        bias_last = tl.load(bias_last_ptr + tl.arange(0, M_last), mask=(tl.arange(0, M_last) < M_last))
        
        # Compute matrix multiplication
        acc_first = tl.dot(x.to(tl.float32), weight_first.to(tl.float32)).to(tl.float32) + bias_first[None, :]
        acc_last = tl.dot(x.to(tl.float32), weight_last.to(tl.float32)).to(tl.float32) + bias_last[None, :]
        
        # Store results
        tl.store(out_first_ptr + (M_offset + tl.arange(0, BLOCK_M))[:, None] * out_first_stride + 
                tl.arange(0, M_first)[None, :], acc_first, mask=M_mask[:, None] & (tl.arange(0, M_first)[None, :] < M_first))
        tl.store(out_last_ptr + (M_offset + tl.arange(0, BLOCK_M))[:, None] * out_last_stride + 
                tl.arange(0, M_last)[None, :], acc_last, mask=M_mask[:, None] & (tl.arange(0, M_last)[None, :] < M_last))

@torch.fx.wrap
def optimized_fusion_linear(in_0, in_1, in_2, in_3, in_4, in_5):
    """Optimized linear operation with column slicing fusion"""
    
    # First linear operation: compute only first and last 256 columns
    N1, K1 = in_5.shape
    M1_first, M1_last = 256, 256
    
    # Create output tensors for first linear operation
    out_first_1 = torch.empty((N1, M1_first), dtype=in_5.dtype, device=in_5.device) 
    out_last_1 = torch.empty((N1, M1_last), dtype=in_5.dtype, device=in_5.device)
    
    # Launch kernel for first linear
    BLOCK_M = 64
    BLOCK_K = 32
    grid_size = (triton.cdiv(N1, BLOCK_M),)
    
    simple_linear_kernel[grid_size](
        in_5, in_5.stride(0),
        in_1, in_1.stride(0), in_1.stride(1),  # weight_first
        in_1, in_1.stride(0), in_1.stride(1),  # weight_last (same weight matrix)
        in_0, in_0,  # bias vectors
        out_first_1, out_first_1.stride(1),
        out_last_1, out_last_1.stride(1),
        N1, K1, M1_first, M1_last,
        BLOCK_M, BLOCK_K
    )
    
    # Reshape outputs for first linear
    tmp_6 = out_first_1.view(-1, M1_first)
    tmp_8 = out_last_1.view(-1, M1_last)
    
    # Second linear operation: reshape input and compute channel-wise
    tmp_9 = in_4.reshape(300, -1, 256)
    N2, N_mid, K2 = tmp_9.shape
    M2_first, M2_last = 256, 256
    
    # Create output tensors for second linear operation
    out_first_2 = torch.empty((N2, N_mid, M2_first), dtype=tmp_9.dtype, device=tmp_9.device)
    out_last_2 = torch.empty((N2, N_mid, M2_last), dtype=tmp_9.dtype, device=tmp_9.device)
    
    # Launch kernel for second linear
    grid_size_2 = (triton.cdiv(N2, BLOCK_M),)
    
    simple_linear_kernel[grid_size_2](
        tmp_9, tmp_9.stride(0), tmp_9.stride(1),
        in_3, in_3.stride(0), in_3.stride(1),  # weight_first
        in_3, in_3.stride(0), in_3.stride(1),  # weight_last (same weight matrix)
        in_2, in_2,  # bias vectors
        out_first_2, out_first_2.stride(1), out_first_2.stride(2),
        out_last_2, out_last_2.stride(1), out_last_2.stride(2),
        N2, K2, M2_first, M2_last,
        BLOCK_M, BLOCK_K
    )
    
    # Final operations
    tmp_13 = tmp_6.unsqueeze(-2)
    
    return out_first_2, out_last_2, tmp_8, tmp_13

def replacement_func():
    return optimized_fusion_linear