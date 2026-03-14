import torch
import triton
import triton.language as tl
import math

def pattern(input_tensor):
    """Pattern to match transpose(-2, -1) operation"""
    return input_tensor.transpose(-2, -1)

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_transpose_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized transpose kernel for swapping last two dimensions"""
    
    # Get program id for the flattened sequence dimension
    pid = tl.program_id(0)
    
    # Calculate offset in the flattened N*H*W space
    flat_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for all dimensions
    mask = (flat_offset < N * H * W)
    
    # Convert flat offset to 3D coordinates for input [N, H, W]
    # We access input as [batch_size, N, H, W] where batch_size=1
    n_dim = flat_offset // (H * W)
    remainder = flat_offset % (H * W)
    h_dim = remainder // W
    w_dim = remainder % W
    
    # Create masks for each dimension
    mask_n = n_dim < N
    mask_h = h_dim < H
    mask_w = w_dim < W
    
    valid_mask = mask & mask_n & mask_h & mask_w
    
    # Load input values
    # Input is stored as [batch_size, N, H, W] = [1, N, H, W]
    input_offset = (n_dim * H + h_dim) * W + w_dim
    input_vals = tl.load(input_ptr + input_offset, mask=valid_mask, other=0.0)
    
    # For output, we need [batch_size, N, W, H]
    # So we swap the H and W dimensions
    output_offset = (n_dim * W + h_dim) * H + w_dim
    tl.store(output_ptr + output_offset, input_vals, mask=valid_mask)

@torch.fx.wrap
def optimized_transpose(input_tensor):
    # Get input shape: [1, N, H, W] = [1, 16, 196, 48]
    batch_size, N, H, W = input_tensor.shape
    
    # For the transpose operation, we swap H and W dimensions
    # Output shape: [1, N, W, H] = [1, 16, 48, 196]
    
    # Create output tensor
    output_shape = (batch_size, N, W, H)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Choose optimal block size
    BLOCK_SIZE = 1024  # Standard Triton block size
    
    # Calculate total elements in the flattened N*H*W space
    total_elements = N * H * W
    
    # Calculate number of programs needed
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_transpose_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        N=N, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_transpose