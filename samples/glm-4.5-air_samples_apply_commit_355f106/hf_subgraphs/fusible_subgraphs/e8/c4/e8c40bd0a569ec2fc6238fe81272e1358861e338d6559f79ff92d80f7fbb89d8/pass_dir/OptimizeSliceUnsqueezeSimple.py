import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x):
    # Match: slice (select index 0 along dim 1) followed by unsqueeze at dim 1
    sliced = x[slice(None, None, None), 0]  # x[:, 0]
    unsqueezed = torch.unsqueeze(sliced, 1)  # Add dimension at position 1
    return unsqueezed

# Argument extraction function
def replacement_args(x):
    return (x,)

# Simple implementation using basic operations - might be faster for small tensors
def slice_unsqueeze_simple(x):
    # Get input tensor properties
    batch_size, old_dim1, dim2, dim3 = x.shape
    
    # For very small tensors, use simple operations instead of Triton kernel
    total_elements = dim2 * dim3
    
    # If tensor is very small, use simple operations
    if total_elements <= 2048:  # Threshold for when simple operations are better
        # Use advanced indexing to directly get the slice and reshape
        # This approach: x[:, 0, :, :] -> then reshape to add dimension
        result = x[:, 0:1, :, :]  # This directly gives us [batch, 1, dim2, dim3]
        return result
    
    # For larger tensors, use Triton kernel
    out_shape = (batch_size, 1, dim2, dim3)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Use smaller block size for better performance
    if total_elements <= 64:
        BLOCK_SIZE = 64
    elif total_elements <= 256:
        BLOCK_SIZE = 128
    elif total_elements <= 1024:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 512
    
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch_size, num_blocks)
    
    # Launch Triton kernel
    slice_unsqueeze_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_batch=batch_size,
        n_old_dim1=old_dim1,
        n_dim2=dim2,
        n_dim3=dim3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

@triton.jit
def slice_unsqueeze_kernel(
    x_ptr,
    out_ptr,
    n_batch,
    n_old_dim1,
    n_dim2,
    n_dim3,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a batch element
    batch_idx = tl.program_id(0)
    element_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = element_idx < (n_dim2 * n_dim3)
    
    # Calculate pointer offsets
    x_base_ptr = x_ptr + batch_idx * n_old_dim1 * n_dim2 * n_dim3
    
    # Load data from first slice only (dim1=0)
    x_flat_offset = element_idx
    x = tl.load(x_base_ptr + x_flat_offset, mask=mask, other=0.0)
    
    # Store result with new dimension structure
    out_flat_offset = batch_idx * n_dim2 * n_dim3 + element_idx
    tl.store(out_ptr + out_flat_offset, x, mask=mask)

@torch.fx.wrap
def slice_unsqueeze_kernel_wrapper(x):
    return slice_unsqueeze_simple(x)

# Replacement function
def replacement_func():
    return slice_unsqueeze_kernel_wrapper