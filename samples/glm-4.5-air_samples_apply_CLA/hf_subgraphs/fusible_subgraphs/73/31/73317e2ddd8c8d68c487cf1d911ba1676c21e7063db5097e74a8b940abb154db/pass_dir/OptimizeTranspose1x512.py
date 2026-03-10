import torch
import triton
import triton.language as tl

def pattern(x):
    # x is expected to be a 1x512 tensor
    transpose_val = x.t()
    return (transpose_val,)

def replacement_args(x):
    return (x,)

@triton.jit
def transpose_1x512_kernel(
    x_ptr,  # input tensor: 1x512
    out_ptr, # output tensor: 512x1
    n_cols: tl.constexpr,
):
    # Each program handles one element
    idx = tl.program_id(0)
    
    # Load from input (scalar load for 1x512)
    x_val = tl.load(x_ptr + idx, mask=idx < n_cols, other=0.0)
    
    # Store to output (direct index for 512x1)
    tl.store(out_ptr + idx, x_val)

@torch.fx.wrap
def optimized_transpose_1x512(x):
    # Input is expected to be 2D tensor
    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {x.dim()}D")
    
    n_rows, n_cols = x.shape
    
    # Create output tensor: [n_cols, n_rows]
    out = torch.empty((n_cols, n_rows), dtype=x.dtype, device=x.device)
    
    # Launch kernel - one program per element for simplicity
    n_elements = n_rows * n_cols
    transpose_1x512_kernel[(n_elements,)](
        x_ptr=x,
        out_ptr=out,
        n_cols=n_cols,
    )
    
    return out

def replacement_func():
    return optimized_transpose_1x512