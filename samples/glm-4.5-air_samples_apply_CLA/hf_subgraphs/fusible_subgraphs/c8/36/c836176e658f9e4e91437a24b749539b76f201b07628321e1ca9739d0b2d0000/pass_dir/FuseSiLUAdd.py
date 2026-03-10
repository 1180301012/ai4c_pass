import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Apply SiLU activation on x (equivalent to x * sigmoid(x))
    silu_x = torch.nn.functional.silu(x, inplace=True)
    # Add the result to y
    result = silu_x + y
    return result

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_silu_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that computes SiLU(x) + y in a single operation"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for out-of-bounds elements
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation: SiLU(x) + y = x * sigmoid(x) + y
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_x = x * sigmoid_x
    output = silu_x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.heuristics({
    'BLOCK_SIZE': lambda args: 2048,
})
@triton.jit
def fused_silu_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that computes SiLU(x) + y = x * sigmoid(x) + y"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for out-of-bounds elements
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU: silu(x) = x * sigmoid(x)
    # Use numerically stable sigmoid computation
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_x = x * sigmoid_x
    
    # Add the second input
    output = silu_x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def fused_silu_add_wrapper(x, y):
    """Wrapper function to launch the fused SiLU + Add kernel"""
    # Determine tensor size and launch configuration
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Launch kernel
    grid = (triton.cdiv(n_elements, 2048),)  # Use block size 2048 as default
    fused_silu_add_kernel[grid](
        x,
        y, 
        output,
        n_elements,
        BLOCK_SIZE=2048
    )
    
    return output

def replacement_func():
    return fused_silu_add_wrapper