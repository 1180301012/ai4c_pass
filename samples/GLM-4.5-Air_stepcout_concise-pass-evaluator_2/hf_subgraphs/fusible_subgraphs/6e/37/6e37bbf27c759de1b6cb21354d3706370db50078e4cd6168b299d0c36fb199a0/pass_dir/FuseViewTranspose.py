import torch
import triton
import triton.language as tl
import math

def pattern(tmp_1, view_shape, dim1, dim2):
    tmp_2 = tmp_1.view(view_shape)
    tmp_3 = tmp_2.transpose(dim1, dim2)
    return tmp_2, tmp_3

def replacement_args(tmp_1, view_shape, dim1, dim2):
    return (tmp_1, view_shape, dim1, dim2)

@triton.jit
def fused_view_transpose_kernel(
    input_ptr,
    output_ptr,
    input_stride_0, input_stride_1, input_stride_2,
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
    block_size_p: tl.constexpr,
    block_size_q: tl.constexpr,
):
    # Get program ID
    m = tl.program_id(0)
    n = tl.program_id(1)
    p = tl.program_id(2)
    q = tl.program_id(3)
    
    # Calculate output coordinate (fused layout)
    out_m = m * block_size_m + tl.arange(0, block_size_m)
    out_n = n * block_size_n + tl.arange(0, block_size_n)
    out_p = p * block_size_p + tl.arange(0, block_size_p)
    out_q = q * block_size_q + tl.arange(0, block_size_q)
    
    # Mask for bounds checking
    mask_m = out_m < 64
    mask_n = out_n < 256  # This is the -1 dimension after view
    mask_p = out_p < 128
    mask_q = out_q < 128
    
    # Convert output coordinate back to input coordinate
    # Input: [64, 128, 512] -> after view: [64, 128, 256, 128] -> after transpose: [64, 256, 128, 128]
    # So output [m, n, p, q] corresponds to input [m, p, n*2 + q//64, q%64]
    # Actually, let's think more carefully:
    # Original after view: [64, 128, 256, 128]
    # After transpose(1,2): [64, 256, 128, 128]
    # So output [m, n, p, q] comes from view [m, p, n, q]
    # And view [m, p, n, q] comes from input [m, p, n*128 + q, :]
    
    # Calculate input coordinate
    input_batch = out_m
    input_seq = out_p[:, None]  # This was dimension 1 in original, becomes dimension 2 after transpose
    input_flat = (out_n[:, None] * 128 + out_q[None, :])  # Combine dimensions 2 and 3 from view
    
    # Create mask for input loading
    input_mask = mask_m[:, None, None] & (input_seq < 128) & (input_flat < 512)
    
    # Load input data
    input_offsets = input_batch[:, None, None] * input_stride_0 + input_seq + input_flat * input_stride_2
    input_data = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
    
    # Store output data (already in correct layout from view+transpose)
    output_offsets = (
        out_m[:, None, None, None] * (256 * 128 * 128) + 
        out_n[:, None, None] * (128 * 128) + 
        out_p[:, None] * 128 + 
        out_q
    )
    
    # Create output mask
    output_mask = mask_m[:, None, None, None] & mask_n[:, None, None] & mask_p[:, None] & mask_q
    
    tl.store(output_ptr + output_offsets, input_data, mask=output_mask)

@torch.fx.wrap
def fused_view_transpose_optimized(tmp_1, view_shape, dim1, dim2):
    # Get input shape
    input_shape = tmp_1.shape
    
    # Calculate output shape after view and transpose
    # For our case: input [B, S, D] -> view [B, S, D1, D2] -> transpose [B, D1, S, D2]
    # where D1*D2 = D
    
    # Extract dimensions from view_shape
    if len(view_shape) == 4:
        B, S, D1, D2 = view_shape
        D = D1 * D2
    else:
        raise ValueError(f"Expected 4D view shape, got {view_shape}")
    
    # Verify input shape matches expectation
    if (input_shape[0] != B or input_shape[1] != S or input_shape[2] != D):
        raise ValueError(f"Input shape {input_shape} doesn't match expected dimensions for view {view_shape}")
    
    # Final output shape after transpose
    output_shape = (B, D1, S, D2)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=tmp_1.dtype, device=tmp_1.device)
    
    # Triton launch parameters
    BLOCK_SIZE_M = 64   # Batch dimension
    BLOCK_SIZE_N = 32   # Transposed seq dimension (was D1)
    BLOCK_SIZE_P = 32   # Original seq dimension
    BLOCK_SIZE_Q = 64   # D2 dimension
    
    # Calculate grid size
    grid_m = (B + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (D1 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_p = (S + BLOCK_SIZE_P - 1) // BLOCK_SIZE_P
    grid_q = (D2 + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    
    # Get strides
    input_stride_0 = tmp_1.stride(0)
    input_stride_1 = tmp_1.stride(1)
    input_stride_2 = tmp_1.stride(2)
    
    # Launch kernel
    fused_view_transpose_kernel[(grid_m, grid_n, grid_p, grid_q)](
        tmp_1,
        output,
        input_stride_0, input_stride_1, input_stride_2,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_Q
    )
    
    return tmp_1, output  # Return original view and transposed result

def replacement_func():
    return fused_view_transpose_optimized