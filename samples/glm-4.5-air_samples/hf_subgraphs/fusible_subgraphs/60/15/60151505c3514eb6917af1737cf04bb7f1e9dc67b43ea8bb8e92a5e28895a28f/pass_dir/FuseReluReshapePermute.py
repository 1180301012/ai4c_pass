import torch
import triton
import triton.language as tl
import math

# Pattern matching function - matches the ReLU + Reshape + Permute sequence
def pattern(in_0, in_1):
    # Apply ReLU to in_1 (inplace)
    relu_out = torch.nn.functional.relu(in_1, inplace=True)
    # Reshape in_0 - pattern matches the original computation
    reshaped_in0 = in_0.reshape(-1, 256, -1)
    # Reshape ReLU output to match the pattern
    reshaped_relu = relu_out.reshape(-1, 256, -1)
    # Permute the reshaped ReLU output - creates [batch_size, H, 256] from [batch_size, 256, H]
    permuted_out = reshaped_relu.permute(0, 2, 1)
    return permuted_out, reshaped_in0

def replacement_args(in_0, in_1):
    # Extract batch size and h_size from input shapes
    batch_size = in_0.shape[0]
    h_size = in_0.shape[2]  # This is the H dimension (21 or 19)
    return (in_0, in_1, batch_size, h_size)

@triton.jit
def fused_relu_reshape_permute_kernel(
    # Input pointers
    in_0_ptr, 
    in_1_ptr,
    # Output pointers  
    out_0_ptr,  # permuted result
    out_1_ptr,  # reshaped in_0
    # Shape information
    batch_size: tl.constexpr,
    h_size: tl.constexpr,  # H dimension (21 or 19)
    # Block size for parallel processing
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Each program handles one batch element  
    if pid >= batch_size:
        return
    
    # Total elements per batch after reshape: batch_size * 256 * h_size
    batch_offset = pid * 256 * h_size
    
    # Initialize offsets for processing this batch
    offsets = batch_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_offset + 256 * h_size
    
    # Process in_0: Reshape operation [batch_size, 256, h_size]
    # Load from original [batch_size, 256, H, 1] layout to [batch_size, 256, h_size]
    for i in range(h_size):
        row_offset = batch_offset + i * 256
        row_offsets = row_offset + tl.arange(0, BLOCK_SIZE)
        row_mask = row_offsets < row_offset + 256
        
        # Load data from original layout (contiguous in memory)
        orig_offset = pid * (256 * h_size) + i * 256
        orig_offsets = orig_offset + tl.arange(0, BLOCK_SIZE)
        orig_mask = orig_offsets < orig_offset + 256
        
        in_0_data = tl.load(in_0_ptr + orig_offsets, mask=orig_mask, other=0.0)
        # Store in reshaped layout
        tl.store(out_1_ptr + row_offsets, in_0_data, mask=row_mask)
    
    # Process in_1: ReLU -> Reshape -> Permute
    for i in range(h_size):
        # Load data from original layout for in_1
        orig_offset = pid * (256 * h_size) + i * 256
        orig_offsets = orig_offset + tl.arange(0, BLOCK_SIZE)
        orig_mask = orig_offsets < orig_offset + 256
        
        in_1_data = tl.load(in_1_ptr + orig_offsets, mask=orig_mask, other=0.0)
        # Apply ReLU
        relu_out = tl.maximum(in_1_data, 0.0)
        # Store in permutation layout [batch_size, h_size, 256]
        # We need to store this as [h_size][256] for the batch element
        permuted_offset = pid * (h_size * 256) + i * 256
        permuted_offsets = permuted_offset + tl.arange(0, BLOCK_SIZE)
        permuted_mask = permuted_offsets < permuted_offset + 256
        tl.store(out_0_ptr + permuted_offsets, relu_out, mask=permuted_mask)

@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1):
    batch_size = in_0.shape[0]
    h_size = in_0.shape[2]  # This is H (21 or 19)
    
    # Create output tensors with appropriate shapes
    # For out_0: permuted result should be [batch_size, H, 256] 
    out_0 = torch.empty((batch_size, h_size, in_0.shape[1]), dtype=in_1.dtype, device=in_1.device)
    
    # For out_1: reshaped in_0 should be [batch_size, 256, H]
    out_1 = torch.empty((batch_size, in_0.shape[1], h_size), dtype=in_0.dtype, device=in_0.device)
    
    # Block size - tune based on typical sizes
    BLOCK_SIZE = 512  # Smaller block size for better utilization
    
    # Number of programs equals batch size (each program handles one batch)
    num_programs = batch_size
    
    # Launch the kernel
    fused_relu_reshape_permute_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_0_ptr=out_0,
        out_1_ptr=out_1,
        batch_size=batch_size,
        h_size=h_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_0, out_1

def replacement_func():
    return fused_kernel_wrapper