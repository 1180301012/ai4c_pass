import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern that matches the entire computation structure for Graph 0:
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

# Optimized kernel for single batch scenario (dim0=1)
@triton.jit
def single_batch_kernel(
    x_ptr,           # in_0 pointer [1, 64, 256]
    y_ptr,           # broadcast_tensor pointer [1, 64, 256]
    out_ptr1_ptr,    # output1 store location
    out_ptr2_ptr,    # output2 store location  
    out_ptr3_ptr,    # output3 store location
    x_dim1,          # 64
    x_dim2,          # 256
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Since dim0=1, we only need 2D grid (batch=1, seq_len, features)
    n = tl.program_id(0)
    k = tl.program_id(1)
    
    # Calculate offsets (dim0 is always 0)
    x_offset = 0 * x_dim1 * x_dim2 + n * x_dim2 + k
    y_offset = 0 * x_dim1 * x_dim2 + n * x_dim2 + k  # same as x_offset
    
    # Load values (no needed broadcasting since shapes match)
    x = tl.load(x_ptr + x_offset, other=0.0)
    y = tl.load(y_ptr + y_offset, other=0.0)
    
    # Add operation
    out_val = x + y
    
    # Store directly in transposed position
    # transpose(0,1) swaps batch (0) and sequence (1) dimensions
    out_offset = n * x_dim1 * x_dim2 + 0 * x_dim2 + k  # batch=0 after transpose
    tl.store(out_ptr1_ptr + out_offset, out_val)
    
    # Same for second output (identical computation)
    tl.store(out_ptr2_ptr + out_offset, out_val)
    
    # Original transpose without addition
    orig_x = tl.load(x_ptr + x_offset, other=0.0)
    orig_out_offset = n * x_dim1 * x_dim2 + 0 * x_dim2 + k
    tl.store(out_ptr3_ptr + orig_out_offset, orig_x)

@torch.fx.wrap  
def optimized_single_batch(in_0, in_1):
    """Optimized kernel for single batch scenario (dim0=1)"""
    # Perform the reshape operation first
    broadcast_tensor = in_1.reshape(1, 64, -1)
    
    # Input shapes
    x_shape = in_0.shape  # [1, 64, 256]
    y_shape = broadcast_tensor.shape  # [1, 64, 256]
    
    # Create output tensors
    out1 = torch.empty_like(in_0)  # tmp_4
    out2 = torch.empty_like(in_0)  # tmp_3
    out3 = torch.empty_like(in_0)  # tmp_5
    
    # Since batch size is 1, we can use more efficient 2D grid
    _, N, K = x_shape  # 1, 64, 256
    
    # Choose block sizes optimized for this smaller scenario
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 64
    
    # Calculate grid size (only need N and K dimensions)
    grid_y = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_x = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Launch kernel with 2D grid (batch dimension is implicit 1)
    single_batch_kernel[(grid_y, grid_x)](
        in_0,
        broadcast_tensor,
        out1,
        out2,
        out3,
        N, K,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    return out1, out2, out3

def replacement_args(in_0, in_1):
    """Extract arguments from matched pattern"""
    return (in_0, in_1)

def replacement_func():
    """Return the optimized function"""
    return optimized_single_batch