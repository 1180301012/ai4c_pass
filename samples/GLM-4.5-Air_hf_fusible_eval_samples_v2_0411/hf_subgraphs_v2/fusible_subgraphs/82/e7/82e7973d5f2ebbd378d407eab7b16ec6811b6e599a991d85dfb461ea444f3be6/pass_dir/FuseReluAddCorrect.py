import torch
import triton
import triton.language as tl

def relu_add_pattern(x, y):
    """
    Pattern matching: ReLU on y, then add to x
    This matches the sequence: tmp_0 = relu(in_1), tmp_1 = tmp_0 + in_0
    """
    relu_result = torch.nn.functional.relu(y, inplace=False)
    add_result = relu_result + x
    return add_result

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_relu_add_kernel(
    x_ptr,           # pointer to input x
    y_ptr,           # pointer to input y  
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
    
    # Apply ReLU to y, then add to x
    relu_y = tl.maximum(y, 0.0)
    out = x + relu_y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_add(x, y):
    """
    Fusion of ReLU + Addition operations
    """
    # Ensure tensors are on the same device and have same shape/dtype
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.dtype == y.dtype, "Input tensors must have the same dtype"
    
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
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