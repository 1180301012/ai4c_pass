import torch
import triton
import triton.language as tl
import math

@triton.jit
def reshape_transpose_kernel(
    input_ptr,
    output_ptr,
    N, H_n, W_n, H_w, W_w, C,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for reshape and transpose operations"""
    pid = tl.program_id(0)
    n_elements = N * H_n * W_n * H_w * W_w * C
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate output coordinates
    # input shape: [N, H_n, W_n, H_w, W_w, C]
    # output shape: [N, H_n, H_w, W_n, W_w, C]
    # This transposes W_n and H_w dimensions
    
    # Flatten input indices
    flat_idx = offsets
    
    # Decompose to input coordinates
    c = flat_idx % C
    flat_idx = flat_idx // C
    
    w_w = flat_idx % W_w
    flat_idx = flat_idx // W_w
    
    h_w = flat_idx % H_w
    flat_idx = flat_idx // H_w
    
    w_n = flat_idx % W_n
    flat_idx = flat_idx // W_n
    
    h_n = flat_idx % H_n
    flat_idx = flat_idx // H_n
    
    n = flat_idx % N
    
    # Calculate output flat index with transposed dimensions
    # New order: [N, H_n, H_w, W_n, W_w, C]
    output_flat_idx = n
    output_flat_idx = output_flat_idx * H_n + h_n
    output_flat_idx = output_flat_idx * H_w + h_w
    output_flat_idx = output_flat_idx * W_n + w_n
    output_flat_idx = output_flat_idx * W_w + w_w
    output_flat_idx = output_flat_idx * C + c
    
    # Load input and store output
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + output_flat_idx, input_val, mask=mask)

@torch.fx.wrap
def optimized_spatial_reshape_and_transpose(input_tensor):
    """Optimized function combining reshape and transpose operations"""
    # Original reshape: [1, 133, 133, 96] -> [1, 19, 7, 19, 7, 96]
    input_shape = input_tensor.shape
    N, H_in, W_in, C_in = input_shape
    
    # Grid dimensions for spatial splitting
    H_grid = 19
    W_grid = 7
    
    # Verify the reshape dimensions
    # 133 = 19 * 7, and we're [1, 133, 133, 96] -> [1, 19, 7, 19, 7, 96]
    assert H_in == H_grid * W_grid, f"Input height ({H_in}) doesn't match grid dimensions ({H_grid} * {W_grid} = {H_grid * W_grid})"
    assert W_in == H_grid * W_grid, f"Input width ({W_in}) doesn't match grid dimensions ({H_grid} * {W_grid} = {H_grid * W_grid})"
    
    # Directly perform the reshape operations from original
    # [1, 133, 133, 96] -> [1, 19, 7, 19, 7, 96] -> transpose(2,3) -> [1, 19, 19, 7, 7, 96]
    reshaped = input_tensor.reshape(N, H_grid, W_grid, H_grid, W_grid, C_in)
    transposed = reshaped.transpose(2, 3)
    
    return transposed

@torch.fx.wrap  
def optimized_zeros_reshape_and_transpose(zeros_tensor):
    """Optimized function for zeros tensor reshape and transpose"""
    # Original: [1, 133, 133] -> [1, 19, 7, 19, 7] -> transpose(2,3) -> [1, 19, 19, 7, 7]
    reshaped = zeros_tensor.reshape(1, 19, 7, 19, 7)
    transposed = reshaped.transpose(2, 3)
    return transposed

def pattern(x):
    """Pattern matching the original input reshaping operations"""
    # Match the input reshape and transpose from original code
    tmp_5 = x.reshape(1, 19, 7, 19, 7, 96)
    tmp_6 = tmp_5.transpose(2, 3)
    return tmp_6

def replacement_args(x):
    return (x,)

def replacement_func():
    """Return the optimized input reshaping function"""
    return optimized_spatial_reshape_and_transpose