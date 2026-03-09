import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern that matches the entire computation structure:
    tmp_0 = in_1.reshape(1, 64, -1)
    tmp_1 = in_0 + tmp_0
    tmp_2 = in_0 + tmp_0
    tmp_3 = tmp_1.transpose(0, 1)
    tmp_4 = tmp_2.transpose(0, 1)
    tmp_5 = in_0.transpose(0, 1)
    return (tmp_4, tmp_3, tmp_5)
    """
    # Match the specific reshape operation
    tmp_0 = in_1.reshape(1, 64, -1)
    
    # First broadcast-add -> transpose
    tmp_1 = in_0 + tmp_0
    tmp_3 = tmp_1.transpose(0, 1)
    
    # Second identical broadcast-add -> transpose (redundant)
    tmp_2 = in_0 + tmp_0
    tmp_4 = tmp_2.transpose(0, 1)
    
    # Also add the original input transpose
    tmp_5 = in_0.transpose(0, 1)
    
    return tmp_4, tmp_3, tmp_5

# Optimized kernel that performs broadcast-add + transpose fusion
@triton.jit
def fused_broadcast_add_transpose_kernel(
    x_ptr,           # in_0 pointer
    y_ptr,           # broadcast_tensor pointer [1, 64, 256]
    out_ptr1_ptr,    # pointer where to store first output ptr
    out_ptr2_ptr,    # pointer where to store second output ptr
    out_ptr3_ptr,    # pointer where to store third output ptr
    x_dim0,          # 64
    x_dim1,          # 64
    x_dim2,          # 256
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Grid: (batch_size, seq_len, features)
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.program_id(2)
    
    # Calculate offsets
    x_offset = m * x_dim1 * x_dim2 + n * x_dim2 + k
    y_offset = 0 * x_dim1 * x_dim2 + n * x_dim2 + k  # y has dim0=1
    
    # Load values with broadcasting
    x = tl.load(x_ptr + x_offset, other=0.0)
    y = tl.load(y_ptr + y_offset, other=0.0)
    
    # Add operation
    out_val = x + y
    
    # Store directly in transposed position
    # transpose(0,1) means we swap M and N dimensions
    out_offset_m = n * x_dim0 * x_dim2 + m * x_dim2 + k
    tl.store(out_ptr1_ptr + out_offset_m, out_val)
    
    # Same for second output (identical computation)
    tl.store(out_ptr2_ptr + out_offset_m, out_val)
    
    # Original transpose without addition
    orig_x = tl.load(x_ptr + x_offset, other=0.0) 
    orig_out_offset = n * x_dim0 * x_dim2 + m * x_dim2 + k
    tl.store(out_ptr3_ptr + orig_out_offset, orig_x)

@torch.fx.wrap
def fused_broadcast_add_transpose(in_0, in_1):
    """Optimized fused operation for reshape + broadcast-add + transpose"""
    # Perform the reshape operation first
    broadcast_tensor = in_1.reshape(1, 64, -1)
    
    # Input shapes
    x_shape = in_0.shape  # [64, 64, 256]
    y_shape = broadcast_tensor.shape  # [1, 64, 256]
    
    # Create output tensors
    # tmp_3, tmp_4: transposed result of [in_0 + broadcast_tensor] = [64, 64, 256]
    # tmp_5: transposed in_0 = [64, 64, 256]
    out1 = torch.empty_like(in_0)  # tmp_4
    out2 = torch.empty_like(in_0)  # tmp_3  
    out3 = torch.empty_like(in_0)  # tmp_5
    
    # Triton kernel launch configuration
    M, N, K = x_shape  # 64, 64, 256
    
    # Choose block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 64
    
    # Calculate grid size
    grid_z = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_y = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N  
    grid_x = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Launch kernel
    fused_broadcast_add_transpose_kernel[(grid_z, grid_y, grid_x)](
        in_0,
        broadcast_tensor,
        out1,
        out2, 
        out3,
        M, N, K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    
    return out1, out2, out3

def replacement_args(in_0, in_1):
    """Extract arguments from matched pattern"""
    return (in_0, in_1)

def replacement_func():
    """Return the optimized function"""
    return fused_broadcast_add_transpose