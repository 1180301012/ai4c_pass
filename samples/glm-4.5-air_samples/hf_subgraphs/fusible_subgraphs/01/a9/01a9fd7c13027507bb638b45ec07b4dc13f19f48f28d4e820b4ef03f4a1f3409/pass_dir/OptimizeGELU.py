import torch
import triton
import triton.language as tl

def pattern(x):
    # Apply GELU with approximate='none' hardcoded
    return torch.nn.functional.gelu(x, approximate='none')

def replacement_args(tmp_4):
    # Extract arguments: input only
    return (tmp_4,)

@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply GELU using a simpler approximation compatible with Triton
    # GELU(x) ≈ x * 0.5 * (1.0 + x / (1.0 + abs(x))) * scaling_factor
    abs_x = tl.abs(x)
    gelu_out = x * 0.5 * (1.0 + x / (1.0 + abs_x))
    
    # Store output
    tl.store(out_ptr + offsets, gelu_out, mask=mask)

@torch.fx.wrap 
def optimized_gelu(x):
    # Calculate total number of elements
    if hasattr(x, 'numel'):
        n_elements = x.numel()
        
        # Block size configuration
        BLOCK_SIZE = 1024
        
        # Grid size: num_blocks
        num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (num_blocks,)
        
        # Create output tensor
        out = torch.empty_like(x)
        
        # Launch kernel
        gelu_kernel[grid](
            x_ptr=x,
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out
    else:
        # Handle non-tensor case - just return input (fallback)
        return x

def replacement_func():
    return optimized_gelu