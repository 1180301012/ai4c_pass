import torch
import triton
import triton.language as tl

def pattern(x, y):
    tmp_0 = torch.nn.functional.silu(x, inplace=False)
    tmp_1 = tmp_0 * y
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False) 
    return tmp_2

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_silu_multiply_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Fused SILU multiplication kernel: out = x * sigmoid(x) * y"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation: x * sigmoid(x) * y
    # Using stable sigmoid computation with overflow protection
    sigmoid_x = 1.0 / (1.0 + tl.exp(-tl.where(x > 0, x, 0.0)))
    silu_x = x * sigmoid_x
    out = silu_x * y
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_silu_multiply(x, y):
    """Fused SILU multiplication with no-op dropout removal"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Optimized for typical GPU architectures
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_silu_multiply_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_silu_multiply