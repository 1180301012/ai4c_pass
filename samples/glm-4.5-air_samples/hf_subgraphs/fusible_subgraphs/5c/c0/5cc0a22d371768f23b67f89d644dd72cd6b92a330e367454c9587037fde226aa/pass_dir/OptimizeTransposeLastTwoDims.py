import torch
import triton
import triton.language as tl

# Pattern matching function - matches transpose of last two dimensions
def pattern(x):
    # Transpose last two dimensions: [a, b, c, d] -> [a, b, d, c]
    return x.transpose(-2, -1)


# Argument extraction function
def replacement_args(x):
    return (x,)


# Optimized kernel for transpose of last two dimensions
@triton.jit
def transpose_last_two_kernel(
    x_ptr,
    out_ptr,
    n_batch, n_dim1, n_dim2, n_dim3,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    """
    Optimized kernel to transpose the last two dimensions
    Input:  [batch, dim1, dim2, dim3] -> Output: [batch, dim1, dim3, dim2]
    """
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Determine ranges for this program
    m_start = pid_m * BLOCK_SIZE_X
    n_start = pid_n * BLOCK_SIZE_Y
    
    # Create offsets
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_X)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_Y)
    
    # Create masks
    m_mask = m_offsets < n_batch
    n_mask = n_offsets < n_dim1
    
    # Calculate flat indices for input and output tensors
    # Input strides: [batch, dim1, dim2, dim3]
    x_stride_batch = n_dim1 * n_dim2 * n_dim3
    x_stride_dim1 = n_dim2 * n_dim3
    x_stride_dim3 = n_dim3
    
    # Output strides: [batch, dim1, dim3, dim2] 
    out_stride_batch = n_dim1 * n_dim3 * n_dim2
    out_stride_dim1 = n_dim3 * n_dim2
    out_stride_dim3 = n_dim2
    
    # Process each block efficiently
    for b_idx in range(m_offsets.numel()):
        if not m_mask[b_idx]:
            continue
            
        for d1_idx in range(n_offsets.numel()):
            if not n_mask[d1_idx]:
                continue
                
            # Load the entire block for this batch and dim1 combination
            # Input: [batch, dim1, dim2, dim3] -> we transpose dim2 and dim3
            for d2 in range(0, n_dim2, BLOCK_SIZE_Y):
                d3_end = min(d2 + BLOCK_SIZE_Y, n_dim3)
                
                # Load input block: [dim2, dim3]
                x_start = (m_offsets[b_idx] * x_stride_batch + 
                          n_offsets[d1_idx] * x_stride_dim1 + 
                          d2)
                x_block = tl.load(x_ptr + x_start + tl.arange(0, d3_end - d2) * x_stride_dim3,
                                mask=(tl.arange(0, d3_end - d2) < (d3_end - d2)), other=0.0)
                
                # Store in transposed position: [dim3, dim2]
                out_start = (m_offsets[b_idx] * out_stride_batch + 
                           n_offsets[d1_idx] * out_stride_dim1 + 
                           (d3_end - d2 - 1) * out_stride_dim3)
                
                # Store in reverse order to achieve transpose
                for i, val in enumerate(x_block):
                    out_pos = out_start - i * out_stride_dim3
                    if d3_end - d2 > 0:
                        tl.store(out_ptr + out_pos, val, other=0.0)

@torch.fx.wrap
def optimized_transpose(x):
    """Optimized transpose of last two dimensions"""
    batch, dim1, dim2, dim3 = x.shape
    
    # Choose block sizes based on typical GPU thread organization
    BLOCK_SIZE_X = max(1, min(batch, 32))      # Batch dimension blocking
    BLOCK_SIZE_Y = max(1, min(dim1, 16))      # First dimension blocking
    
    # Calculate grid dimensions
    batch_grid = (batch + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    dim1_grid = (dim1 + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    
    # Create output tensor with transposed dimensions
    output = torch.empty((batch, dim1, dim3, dim2), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    transpose_last_two_kernel[(batch_grid, dim1_grid)](
        x_ptr=x,
        out_ptr=output,
        n_batch=batch,
        n_dim1=dim1,
        n_dim2=dim2,
        n_dim3=dim3,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )
    
    return output

def replacement_func():
    return optimized_transpose