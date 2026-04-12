import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matching: ReLU + Addition
    """
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_relu_add_kernel(
    x_ptr,           # in_0 pointer
    y_ptr,           # in_1 pointer
    out_ptr,         # output pointer  
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