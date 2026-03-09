import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern that matches the entire computation structure for Graph 7:
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

# Optimized kernel for large tensor scenario with memory coalescing
@triton.jit
def large_tensor_kernel(
    x_ptr,           # in_0 pointer [256, 64, 256]
    y_ptr,           # broadcast_tensor pointer [1, 64, 256]
    out_ptr1_ptr,    # output1 store location
    out_ptr2_ptr,    # output2 store location
    out_ptr3_ptr,    # output3 store location
    x_dim0,          # 256 (large batch dimension)
    x_dim1,          # 64
    x_dim2,          # 256
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Grid: (batch_size, seq_len)
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Create shared memory for this workgroup for better cache reuse
    x_shared = tl.zeros((BLOCK_SIZE_M, x_dim2), dtype=tl.float32)
    y_shared = tl.zeros((BLOCK_SIZE_M, x_dim2), dtype=tl.float32)
    
    # Load x data (with potential out-of-bound checks)
    for i in range(BLOCK_SIZE_M):
        m_idx = m * BLOCK_SIZE_M + i
        if m_idx < x_dim0:
            # Load entire row for this batch position
            for k in range(0, x_dim2, 4):  # Vectorized load
                offset = m_idx * x_dim1 * x_dim2 + n * x_dim2 + k
                if k + 3 < x_dim2:
                    vec = tl.load(x_ptr + offset + tl.arange(0, 4), other=0.0)
                else:
                    scalar = tl.load(x_ptr + offset, other=0.0)
                    vec = tl.full((4,), scalar, dtype=tl.float32)
                x_shared[i, k:k+4] = vec
        else:
            x_shared[i, :] = 0.0
    
    # Load y data (broadcasting - same y for all m_idx)
    for i in range(BLOCK_SIZE_M):
        for k in range(0, x_dim2, 4):  # Vectorized load
            offset = 0 * x_dim1 * x_dim2 + n * x_dim2 + k  # y dim0=1
            if k + 3 < x_dim2:
                vec = tl.load(y_ptr + offset + tl.arange(0, 4), other=0.0)
            else:
                scalar = tl.load(y_ptr + offset, other=0.0)
                vec = tl.full((4,), scalar, dtype=tl.float32)
            y_shared[i, k:k+4] = vec
    
    # Compute addition and store in transposed format
    for i in range(BLOCK_SIZE_M):
        m_idx = m * BLOCK_SIZE_M + i
        if m_idx < x_dim0:
            # Add corresponding rows
            add_result = x_shared[i, :] + y_shared[i, :]
            
            # Store in transposed position (swapped m and n)
            out_offset = n * x_dim0 * x_dim2 + m_idx * x_dim2 + tl.arange(0, x_dim2)
            
            # Store both outputs (they're identical)
            tl.store(out_ptr1_ptr + out_offset, add_result)
            tl.store(out_ptr2_ptr + out_offset, add_result)
            
            # Store original transpose (no addition)
            tl.store(out_ptr3_ptr + out_offset, x_shared[i, :])

@torch.fx.wrap
def optimized_large_tensor(in_0, in_1):
    """Optimized kernel for large tensor scenario with memory coalescing"""
    # Perform the reshape operation first
    broadcast_tensor = in_1.reshape(1, 64, -1)
    
    # Input shapes
    x_shape = in_0.shape  # [256, 64, 256]
    y_shape = broadcast_tensor.shape  # [1, 64, 256]
    
    # Create output tensors
    out1 = torch.empty_like(in_0)  # tmp_4
    out2 = torch.empty_like(in_0)  # tmp_3
    out3 = torch.empty_like(in_0)  # tmp_5
    
    M, N, K = x_shape  # 256, 64, 256
    
    # Choose block sizes optimized for large batch dimension
    BLOCK_SIZE_M = 32  # Larger block for big batch dimension
    BLOCK_SIZE_N = 16  # Medium block for sequence dimension
    
    # Calculate grid size
    grid_z = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_y = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel with shared memory optimizations
    large_tensor_kernel[(grid_z, grid_y)](
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
    return optimized_large_tensor