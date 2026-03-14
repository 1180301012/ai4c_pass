import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """
    Simple pattern: reshape followed by permute - test basic matching
    """
    # First reshape: [N, C, D, 1] -> [N, C, D]
    t1 = input_tensor.reshape(32, 256, -1)  # Use exact values from model
    # Then permute: [N, C, D] -> [N, D, C]  
    t2 = t1.permute(0, 2, 1)
    return t2

def replacement_args(reshaped_tensor):
    """
    Extract arguments for replacement
    We only need the original tensor for the fused operation
    """
    return (reshaped_tensor,)

@triton.jit
def fused_reshape_permute_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    n_channels,
    n_length,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block in the flattened output space
    program_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = program_idx < n_batch * n_length * n_channels
    
    # Decompose program index to output coordinates [batch, length, channel]
    program_idx_per_elem = program_idx  # This is the position in [B, D, C] output
    channel_idx = program_idx_per_elem % n_channels
    remaining_idx = program_idx_per_elem // n_channels
    length_idx = remaining_idx % n_length
    batch_idx = remaining_idx // n_length
    
    # Input is [B, C, D, 1] and we want to get element [batch_idx, channel_idx, length_idx, 0]
    input_offset = batch_idx * n_channels * n_length + channel_idx * n_length + length_idx
    output_offset = program_idx_per_elem
    
    # Load and store with vectorized memory access
    input_vals = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + output_offset, input_vals, mask=mask)

def kernel_wrapper(input_tensor):
    # Get input shape [N, C, D, 1]
    original_shape = input_tensor.shape
    n_batch, n_channels, n_length, _ = original_shape
    
    # Output shape after permutation: [N, D, C]
    output_shape = (n_batch, n_length, n_channels)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid dimensions - simple 1D grid over all elements
    total_elements = n_batch * n_length * n_channels
    BLOCK_SIZE = 1024  # Optimal block size for good occupancy
    
    # Launch kernel with simple 1D grid
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_reshape_permute_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_batch=n_batch,
        n_channels=n_channels,
        n_length=n_length,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@torch.fx.wrap
def fused_reshape_permute_wrapper(input_tensor):
    return kernel_wrapper(input_tensor)

def replacement_func():
    """
    Return optimized kernel that fuses reshape + permute operations
    This directly computes permuted output from original [N, C, D, 1] input
    """
    return fused_reshape_permute_wrapper