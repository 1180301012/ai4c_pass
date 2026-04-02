import torch
import triton
import triton.language as tl

# Pattern matching function for slicing and addition
def pattern(x, y):
    """Match slicing to 124 elements and addition"""
    x_sliced = x[(Ellipsis, slice(None, 124, None))]
    y_sliced = y[(Ellipsis, slice(None, 124, None))]
    result = x_sliced + y_sliced
    return result

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Optimized kernel: fused slicing + addition
@triton.jit
def fused_slice_add_kernel(
    x_ptr, y_ptr, out_ptr,
    N, C, L,              # Original shapes: [N, C, L]
    slice_len: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    # Get program ids
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    # Calculate output bounds (only up to slice_len)
    if pid_n >= N or pid_c >= C or pid_l >= slice_len:
        return
    
    # Compute input indices
    x_offset = pid_n * C * L + pid_c * L + pid_l
    y_offset = pid_n * C * L + pid_c * L + pid_l  # Same slicing for both tensors
    
    # Load values with masking for safety
    x_val = tl.load(x_ptr + x_offset, other=0.0)
    y_val = tl.load(y_ptr + y_offset, other=0.0)
    
    # Add and store
    result = x_val + y_val
    out_offset = pid_n * C * slice_len + pid_c * slice_len + pid_l
    tl.store(out_ptr + out_offset, result, other=0.0)

@torch.fx.wrap
def fused_slice_add(x, y, slice_len=124):
    """Fused slicing and addition kernel wrapper"""
    # Get tensor shapes (assume both tensors have same shape)
    N, C, L = x.shape
    
    # Create output tensor with sliced dimension
    output_shape = (N, C, slice_len)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Launch grid configuration
    grid = (
        N,
        C,
        slice_len,
    )
    
    # Optimal block sizes for this shape
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C = 64   # Process multiple channels per thread
    BLOCK_SIZE_L = 32   # Process multiple positions per thread
    
    # Launch kernel
    fused_slice_add_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=output,
        N=N, C=C, L=L,
        slice_len=slice_len,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_slice_add