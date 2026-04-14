import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Match ReLU(x) + y sequence that appears in the original computation"""
    relu_result = torch.nn.functional.relu(x, inplace=False)
    add_result = relu_result + y
    return add_result

def replacement_args(x, y):
    """Extract arguments for fused ReLU+Add"""
    return (x, y)

@triton.jit
def fused_relu_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + Add kernel: out = max(0, x) + y"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with vectorization
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation: max(0, x) + y
    relu_x = tl.maximum(x, 0.0)
    out = relu_x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_add(x, y):
    """Wrapper for fused ReLU + Add"""
    # Ensure tensors are on the same device
    if x.device != y.device:
        y = y.to(x.device)
    
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    fused_relu_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the fused ReLU + Add function"""
    return fused_relu_add