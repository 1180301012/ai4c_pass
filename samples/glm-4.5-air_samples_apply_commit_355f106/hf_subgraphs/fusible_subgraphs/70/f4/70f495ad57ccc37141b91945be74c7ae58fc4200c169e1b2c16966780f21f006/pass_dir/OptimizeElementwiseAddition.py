import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def add_kernel_3d(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Linear kernel for element-wise addition
    linear_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < n_elements
    
    # Load inputs and compute addition
    x = tl.load(x_ptr + linear_idx, mask=mask, other=0.0)
    y = tl.load(y_ptr + linear_idx, mask=mask, other=0.0)
    out = x + y
    
    # Store result
    tl.store(out_ptr + linear_idx, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    # Only optimize for larger 3D tensors where the overhead is justified
    if x.dim() == 3 and x.numel() > 100000:  # Only for tensors with >100K elements
        out = torch.empty_like(x)
        n_elements = x.numel()
        
        # Use larger block size for better utilization
        BLOCK_SIZE = 2048
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        add_kernel_3d[(num_programs,)](
            x_ptr=x,
            y_ptr=y,
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
    else:
        # For smaller tensors or 1D tensors, use PyTorch's native add
        return x + y

def replacement_func():
    return triton_add