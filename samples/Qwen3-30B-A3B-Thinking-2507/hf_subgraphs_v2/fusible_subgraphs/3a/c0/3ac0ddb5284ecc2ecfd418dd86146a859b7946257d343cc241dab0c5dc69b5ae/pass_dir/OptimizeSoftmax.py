import torch
import triton
import triton.language as tl


def pattern(x, y):
    z = x + y
    w = z.view(8, 300, 625)
    s = torch.nn.functional.softmax(w, dim = -1)
    t = s.view(1, 8, 300, 625)
    return s, t

def replacement_args(x, y):
    return (x, y)

@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * n_cols
    offsets = tl.arange(0, n_cols)
    x = tl.load(x_ptr + row_start + offsets)
    row_max = tl.max(x)
    x_exp = tl.exp(x - row_max)
    row_sum = tl.sum(x_exp)
    out = x_exp / row_sum
    tl.store(y_ptr + row_start + offsets, out)

@torch.fx.wrap
def softmax_wrapper(x, y):
    # Input: x (in_1: [1,8,300,625]), y (in_0: [1,1,300,625])
    # Step 1: Compute z = x + y (shape [1,8,300,625])
    z = x + y
    
    # Step 2: Reshape for softmax (8, 300, 625)
    n_rows = 8 * 300
    n_cols = 625
    
    # Step 3: Allocate output
    out = torch.empty(8, 300, 625, dtype=z.dtype, device=z.device)
    
    # Step 4: Configure kernel
    BLOCK_SIZE = 256
    grid = (n_rows,)
    
    # Step 5: Launch kernel
    softmax_kernel[grid](
        x_ptr=z, 
        y_ptr=out, 
        n_rows=n_rows, 
        n_cols=n_cols, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 6: Create return values
    s = out  # Already (8, 300, 625)
    t = s.view(1, 8, 300, 625)
    return s, t

def replacement_func():
    return softmax_wrapper