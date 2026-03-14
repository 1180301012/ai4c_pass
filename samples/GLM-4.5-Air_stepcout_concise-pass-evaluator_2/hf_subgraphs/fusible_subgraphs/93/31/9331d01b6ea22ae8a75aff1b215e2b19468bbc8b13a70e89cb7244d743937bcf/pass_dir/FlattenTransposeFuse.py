import torch
import triton
import triton.language as tl

def pattern(x):
    # Matches the sequence: flatten(2) -> transpose(1, 2)
    tmp = x.flatten(2)
    result = tmp.transpose(1, 2)
    return result

def replacement_args(x):
    return (x,)

@triton.jit
def flatten_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    dim1,
    dim2,
    dim3,
    block_size: tl.constexpr
):
    """Kernel that fuses flatten(2) and transpose(1, 2) operations"""
    
    # Global program ID
    pid = tl.program_id(0)
    grid_size = batch_size * dim1 * dim2 * dim3
    
    if pid >= grid_size:
        return
    
    # Calculate flat output coordinates after transpose
    # Original: [batch, dim1, dim2, dim3]
    # Flatten(2): [batch, dim1, dim2*dim3] 
    # Transpose(1,2): [batch, dim2*dim3, dim1]
    output_flat_size = dim1 * dim2 * dim3
    
    # Convert 1D ID to output coordinates: [batch, transposed_dim1, transposed_dim2]
    # where transposed_dim1 = dim2*dim3, transposed_dim2 = dim1
    transposed_dim1 = dim2 * dim3
    transposed_dim2 = dim1
    
    bid = pid // (transposed_dim1 * transposed_dim2)
    h = (pid % (transposed_dim1 * transposed_dim2)) // transposed_dim2
    w = pid % transposed_dim2
    
    # Calculate input coordinates
    # Original: [batch, dim1, dim2, dim3]
    # flatten(2): [batch, dim1, dim2*dim3] 
    # We want: [batch, h (from transposed_dim1), w (from transposed_dim2)]
    # Where h corresponds to (dim2*dim3) and w corresponds to dim1
    
    # Convert h back to original (dim2, dim3) coordinates
    orig_dim2 = h // dim3
    orig_dim3 = h % dim3
    
    # Calculate input and output offsets
    input_offset = bid * dim1 * dim2 * dim3 + w * dim2 * dim3 + orig_dim2 * dim3 + orig_dim3
    output_offset = bid * transposed_dim1 * transposed_dim2 + h * transposed_dim2 + w
    
    # Load input value
    value = tl.load(input_ptr + input_offset, other=0.0)
    
    # Store transposed result
    tl.store(output_ptr + output_offset, value)

@triton.jit
def flatten_transpose_coalesced_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    dim1,
    dim2,
    dim3,
    block_height: tl.constexpr,
    block_width: tl.constexpr
):
    """Optimized kernel with better memory coalescing"""
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    transposed_dim1 = dim2 * dim3
    transposed_dim2 = dim1
    
    num_m = triton.cdiv(batch_size * transposed_dim1, block_height)
    num_n = triton.cdiv(transposed_dim2, block_width)
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_m >= num_m or pid_n >= num_n:
        return
    
    # Calculate work tile boundaries
    m_start = pid_m * block_height
    m_end = min((pid_m + 1) * block_height, batch_size * transposed_dim1)
    n_start = pid_n * block_width
    n_end = min((pid_n + 1) * block_width, transposed_dim2)
    
    # Process each element in the work tile
    for bi in range(m_start, m_end):
        for bj in range(n_start, n_end):
            # Convert batch*transposed_dim1 to batch and h coordinates
            bid = bi // transposed_dim1
            h = bi % transposed_dim1
            
            # Convert h back to original (dim2, dim3) coordinates
            orig_dim2 = h // dim3
            orig_dim3 = h % dim3
            
            # Calculate input and output offsets
            input_offset = bid * dim1 * dim2 * dim3 + bj * dim2 * dim3 + orig_dim2 * dim3 + orig_dim3
            output_offset = bid * transposed_dim1 * transposed_dim2 + h * transposed_dim2 + bj
            
            # Load input value
            value = tl.load(input_ptr + input_offset, other=0.0)
            
            # Store transposed result
            tl.store(output_ptr + output_offset, value)

@torch.fx.wrap  
def flattened_transposed(x):
    """Fused flatten(2) + transpose(1, 2) function"""
    batch_size, dim1, dim2, dim3 = x.shape
    
    # Final shape after flatten(2) and transpose(1, 2): [batch_size, dim2*dim3, dim1]
    output_shape = (batch_size, dim2 * dim3, dim1)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Launch appropriate kernel based on tensor dimensions
    total_elements = batch_size * dim2 * dim3 * dim1
    
    if total_elements > 1024 * 1024:  # Large tensors use 2D tiling
        grid = (
            triton.cdiv(batch_size * (dim2 * dim3), 64),
            triton.cdiv(dim1, 64)
        )
        flatten_transpose_coalesced_kernel[grid](
            x, output,
            batch_size, dim1, dim2, dim3,
            64, 64
        )
    else:  # Smaller tensors use 1D grid
        grid = triton.cdiv(total_elements, 256)
        flatten_transpose_kernel[grid](
            x, output,
            batch_size, dim1, dim2, dim3,
            256
        )
    
    return output

def replacement_func():
    return flattened_transposed