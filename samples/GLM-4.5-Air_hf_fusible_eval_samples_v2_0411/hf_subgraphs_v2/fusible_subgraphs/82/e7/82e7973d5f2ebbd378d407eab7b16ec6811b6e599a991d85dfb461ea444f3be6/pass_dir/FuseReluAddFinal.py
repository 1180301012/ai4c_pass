import torch
import triton
import triton.language as tl

def pattern(a, b):
    """
    Pattern matching: targets addition operations that can be fused with preceding ReLU
    The framework will identify the correct addition operation in the context of the full graph
    """
    return a + b

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_relu_add_kernel(
    x_ptr,           # pointer to input x (in_0 in original computation)
    y_ptr,           # pointer to input y (in_1 in original computation)  
    out_ptr,         # pointer to output
    n_elements,      # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU to y, then add to x (matching: relu(in_1) + in_0)
    relu_y = tl.maximum(y, 0.0)
    out = x + relu_y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_add(x, y):
    """
    Fusion of ReLU on y followed by addition with x
    This matches: torch.relu(y) + x
    """
    # Ensure tensors are on the same device and have same shape/dtype
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.dtype == y.dtype, "Input tensors must have the same dtype"
    
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_relu_add_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_relu_add