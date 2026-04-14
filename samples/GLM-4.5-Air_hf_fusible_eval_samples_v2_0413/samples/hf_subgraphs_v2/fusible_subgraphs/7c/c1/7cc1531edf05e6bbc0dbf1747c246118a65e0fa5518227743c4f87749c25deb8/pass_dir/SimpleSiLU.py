import torch
import triton
import triton.language as tl

def pattern(tmp_6):
    # Simple SiLU activation
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace = True)
    return tmp_7

def replacement_args(tmp_6):
    return (tmp_6,)

@triton.jit
def simple_silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    n_programs = tl.cdiv(n_elements, BLOCK_SIZE)
    
    if pid >= n_programs:
        return
    
    # Calculate block bounds
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, n_elements)
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Load input tensor for this block
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Apply SiLU activation: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_out = x * sigmoid_x
    
    # Store output
    tl.store(out_ptr + offsets, silu_out, mask=mask)

@torch.fx.wrap
def simple_silu(x):
    # Get tensor shape
    n_elements = x.numel()
    
    # Use optimal block size
    BLOCK_SIZE = 1024
    
    # Number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    simple_silu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return simple_silu