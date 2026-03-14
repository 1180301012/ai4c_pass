import torch
import triton
import triton.language as tl

def pattern(x, y):
    tmp_0 = torch.nn.functional.silu(x, inplace=False)
    tmp_1 = tmp_0 * y
    return tmp_1

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_silu_multiply_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Fused SILU + multiplication kernel: out = x * sigmoid(x) * y"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with cache hints
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, volatile=False)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0, volatile=False)
    
    # Compute SILU: x * sigmoid(x)
    # Use stable sigmoid computation
    sigmoid_x = 1.0 / (1.0 + tl.exp(-tl.where(x > 0, x, 0.0)))
    silu_x = x * sigmoid_x
    
    # Multiply by y
    out = silu_x * y
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_silu_multiply(x, y):
    """Fused SILU and multiplication to eliminate kernel launch overhead"""
    n_elements = x.numel()
    BLOCK_SIZE = 4096  # Even larger block size for better efficiency
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    
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