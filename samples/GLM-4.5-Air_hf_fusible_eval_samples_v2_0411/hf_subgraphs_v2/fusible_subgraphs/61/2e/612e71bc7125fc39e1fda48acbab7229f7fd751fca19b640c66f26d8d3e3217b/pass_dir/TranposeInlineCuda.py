import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(
    x_ptr,
    out_ptr,
    src_stride_0,
    src_stride_1,
    dst_stride_0,
    dst_stride_1,
    n_src_dim0,
    n_src_dim1,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance transpose kernel for 2D matrices"""
    
    # Each program handles one element in the transposed output
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Check bounds
    if (row_idx >= n_src_dim1) or (col_idx >= n_src_dim0):
        return
    
    # Calculate input and output positions
    src_pos = col_idx * src_stride_0 + row_idx * src_stride_1
    dst_pos = row_idx * dst_stride_0 + col_idx * dst_stride_1
    
    # Load from source and store to transposed position
    val = tl.load(x_ptr + src_pos)
    tl.store(out_ptr + dst_pos, val)

@torch.fx.wrap
def transpose_triton(x):
    """High-performance transpose function"""
    src_dim0, src_dim1 = x.shape
    
    # Create output with transposed dimensions
    out = torch.empty((src_dim1, src_dim0), dtype=x.dtype, device=x.device)
    
    # Handle the case where input is already on GPU
    if x.is_cuda:
        input_tensor = x
    else:
        input_tensor = x.to(device='cuda')
        out = out.to(device='cuda')
    
    src_stride_0, src_stride_1 = x.stride()
    dst_stride_0, dst_stride_1 = out.stride()
    
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions (one program per output element)
    # For GPU efficiency, we limit grid size and use smaller blocks
    max_rows = min(src_dim1, 1024)  # Limit rows for GPU efficiency
    max_cols = min(src_dim0, 1024)  # Limit cols for GPU efficiency
    
    # Launch kernel with optimized grid size
    transpose_kernel[(max_rows, max_cols)](
        x_ptr=input_tensor,
        out_ptr=out,
        src_stride_0=src_stride_0,
        src_stride_1=src_stride_1,
        dst_stride_0=dst_stride_0,
        dst_stride_1=dst_stride_1,
        n_src_dim0=src_dim0,
        n_src_dim1=src_dim1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(x):
    """Match transpose pattern"""
    tmp_2 = x.t()
    return tmp_2  # Return the observable intermediate

def replacement_args(x):
    """Extract argument for replacement"""
    return (x,)

def replacement_func():
    """Return the optimized kernel wrapper"""
    return transpose_triton