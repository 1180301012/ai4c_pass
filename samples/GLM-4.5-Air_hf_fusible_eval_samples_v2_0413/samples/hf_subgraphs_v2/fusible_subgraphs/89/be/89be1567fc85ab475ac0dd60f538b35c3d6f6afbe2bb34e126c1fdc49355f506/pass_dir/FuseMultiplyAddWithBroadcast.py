import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # The exact computation pattern from model.py
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_C': 128}),
        triton.Config({'BLOCK_SIZE_C': 64}),
        triton.Config({'BLOCK_SIZE_C': 256}),
    ],
    key=['B', 'H', 'C'],
)
@triton.jit
def fused_multiply_add_kernel(
    in_0_ptr, 
    in_1_ptr, 
    in_2_ptr, 
    out_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """Fused kernel for in_2 * in_1 + in_0 with broadcasting"""
    # Calculate program indices based on 4D tensor shape [B, H, W, C]
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Each program handles BLOCK_SIZE_C elements in C dimension
    c_offset = pid_c * BLOCK_SIZE_C
    c_indices = tl.arange(0, BLOCK_SIZE_C)
    mask = c_indices < (C - c_offset)  # Mask for remaining elements
    
    # Load in_2: [B, H, 1, C]
    in_2_offset = pid_b * (H * 1 * C) + pid_h * (1 * C) + c_offset
    in_2_data = tl.load(in_2_ptr + in_2_offset + c_indices, mask=mask, other=0.0)
    
    # Load in_1: we need to handle broadcasting from [1,1,2,C] to [B,H,2,C]
    # Extract the [2,C] part from in_1
    # For W=0 slice
    in_1_slice0_offset = 0 * C + c_offset  # First slice of the 2 slices
    in_1_slice0 = tl.load(in_1_ptr + in_1_slice0_offset + c_indices, mask=mask, other=0.0)
    
    # For W=1 slice  
    in_1_slice1_offset = 1 * C + c_offset  # Second slice of the 2 slices
    in_1_slice1 = tl.load(in_1_ptr + in_1_slice1_offset + c_indices, mask=mask, other=0.0)
    
    # Load in_0: [2, C] - handle broadcasting to [B,H,2,C]
    in_0_slice0_offset = 0 * C + c_offset  # First row
    in_0_slice0 = tl.load(in_0_ptr + in_0_slice0_offset + c_indices, mask=mask, other=0.0)
    
    in_0_slice1_offset = 1 * C + c_offset  # Second row  
    in_0_slice1 = tl.load(in_0_ptr + in_0_slice1_offset + c_indices, mask=mask, other=0.0)
    
    # Perform fused computation for both W slices
    # W=0: in_2 * in_1_slice0 + in_0_slice0
    result_slice0 = in_2_data * in_1_slice0 + in_0_slice0
    
    # W=1: in_2 * in_1_slice1 + in_0_slice1 (in_2 broadcasts to W dimension)
    result_slice1 = in_2_data * in_1_slice1 + in_0_slice1
    
    # Store both results in the same output tensor [B, H, 2, C]
    # W=0 slice at offset 0
    out_offset_w0 = pid_b * (H * 2 * C) + pid_h * (2 * C) + 0 * C + c_offset + c_indices
    tl.store(out_ptr + out_offset_w0, result_slice0, mask=mask)
    
    # W=1 slice at offset C
    out_offset_w1 = pid_b * (H * 2 * C) + pid_h * (2 * C) + 1 * C + c_offset + c_indices
    tl.store(out_ptr + out_offset_w1, result_slice1, mask=mask)

@torch.fx.wrap
def fused_multiply_add(in_0, in_1, in_2):
    # Get input shapes
    B = in_2.shape[0]  # Batch size from in_2
    H = in_2.shape[1]  # 17 from weight_meta
    W_out = 2  # Output W dimension from broadcasting with in_1 [1,1,2,C]
    C = in_2.shape[3]  # 128 from weight_meta
    
    # Create output tensor with correct shape [B, H, 2, C]
    out = torch.empty((B, H, W_out, C), dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel with autotuning
    # Grid: [B, H, C//BLOCK_SIZE_C]
    grid = lambda meta: (meta['B'], meta['H'], (meta['C'] + meta['BLOCK_SIZE_C'] - 1) // meta['BLOCK_SIZE_C'])
    
    fused_multiply_add_kernel[grid](
        in_0, in_1, in_2, out,
        B=B, H=H, W=W_out, C=C
    )
    
    return out

def replacement_func():
    return fused_multiply_add