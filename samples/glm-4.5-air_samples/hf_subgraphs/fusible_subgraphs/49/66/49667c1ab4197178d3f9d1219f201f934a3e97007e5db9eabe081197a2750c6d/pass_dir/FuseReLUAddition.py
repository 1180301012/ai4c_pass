import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    return (tmp_1,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def relu_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Fused ReLU + Addition
    out = tl.maximum(x, 0.0) + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_add(x, y):
    # Determine total number of elements
    N = x.numel()
    
    # Use BLOCK_SIZE and grid size optimized for tensor operations
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    relu_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_relu_add