import torch
import triton
import triton.language as tl

# Pattern matching function - matches transpose + device transfer
def pattern(in_0):
    tmp_2 = in_0.t()
    tmp_3 = tmp_2.to(device(type='cuda'))
    return tmp_3

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized transpose kernel using Triton
@triton.jit
def transpose_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # For small matrices, use direct vectorized transpose
    if n_rows == 1 and n_cols > 1:
        # Special case for row vector to column vector
        i = tl.program_id(0)
        if i < n_cols:
            x = tl.load(x_ptr + i)
            tl.store(out_ptr + i * n_rows, x)
    else:
        # General case with vectorized loads
        i = tl.program_id(0)
        j = tl.program_id(1)
        
        if i < n_rows and j < n_cols:
            # Load one element at a time for small matrices to avoid complex indexing
            x = tl.load(x_ptr + i * n_cols + j)
            tl.store(out_ptr + j * n_rows + i, x)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)  
@torch.fx.wrap
def triton_transpose(x):
    n_rows, n_cols = x.shape
    out = torch.empty((n_cols, n_rows), dtype=x.dtype, device=x.device)
    
    # For small matrices, use simpler grid setup
    if n_rows == 1 and n_cols > 1:
        # Row vector to column vector - simplify to 1D grid
        BLOCK_SIZE = 32
        num_programs = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
        transpose_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            n_rows=n_rows,
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # General 2D case
        BLOCK_SIZE = 32
        num_m = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_n = (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
        transpose_kernel[(num_m, num_n)](
            x_ptr=x,
            out_ptr=out,
            n_rows=n_rows,
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return out

# Replacement function
def replacement_func():
    return triton_transpose