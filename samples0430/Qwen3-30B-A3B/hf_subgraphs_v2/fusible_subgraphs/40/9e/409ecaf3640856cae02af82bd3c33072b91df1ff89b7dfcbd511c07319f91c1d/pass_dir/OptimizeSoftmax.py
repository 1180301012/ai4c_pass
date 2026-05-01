import torch
import triton
import triton.language as tl

# Pattern matching function to match multiply and softmax
@torch.fx.wrap
def pattern(x):
    alpha = 0.1767766952966369
    tmp = x * alpha
    result = tmp.softmax(dim=-1)
    return result

# Extract arguments needed for replacement
@torch.fx.wrap
def replacement_args(x):
    return (x,)

# Triton kernel for optimized softmax
@triton.jit
@torch.fx.wrap
def softmax_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # Each program handles a row of the matrix
    row = tl.program_id(0)
    row_start = row * n_cols
    
    # Load the row from input
    x = tl.load(x_ptr + row_start, mask=tl.arange(0, n_cols) < n_cols, other=0.0)
    
    # Compute row max for stability
    row_max = tl.max(x)
    
    # Subtract max and compute exponential
    x = x - row_max
    x = tl.exp(x)
    
    # Compute sum of exponentials
    x_sum = tl.sum(x)
    
    # Compute softmax
    out = x / x_sum
    
    # Store result
    tl.store(out_ptr + row_start, out)

# Wrapper function to launch the kernel
@torch.fx.wrap
def softmax_wrapper(x):
    # Multiply input by the scalar
    x = x * 0.1767766952966369
    # Get tensor dimensions
    B, H, S, _ = x.shape
    n_rows = B * H * S
    n_cols = S
    
    # Create output tensor with same shape
    out = torch.empty_like(x)
    
    # Set block size (empirically tuned for S=400)
    BLOCK_SIZE = 512
    
    # Launch kernel
    num_programs = (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    softmax_kernel[(num_programs,)](
        x_ptr=x, 
        out_ptr=out, 
        n_rows=n_rows, 
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
@torch.fx.wrap
def replacement_func():
    return softmax_wrapper