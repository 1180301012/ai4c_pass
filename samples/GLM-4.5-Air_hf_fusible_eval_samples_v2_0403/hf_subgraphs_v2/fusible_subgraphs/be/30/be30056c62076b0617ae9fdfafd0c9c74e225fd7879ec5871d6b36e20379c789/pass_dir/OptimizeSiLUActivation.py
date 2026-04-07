import torch
import triton
import triton.language as tl

@torch.fx.wrap
def optimized_silu(input_tensor):
    """Main wrapper for optimized SiLU activation"""
    # Get total number of elements
    n_elements = input_tensor.numel()
    
    # Create output tensor (using empty_like to match original behavior)
    output = torch.empty_like(input_tensor)
    
    # Set up grid and block size
    BLOCK_SIZE = 1024  # Optimal block size for GPU
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    silu_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@triton.jit
def silu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized SiLU kernel using Triton"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    # Use stabilized sigmoid computation
    neg_x = -x
    # Clamp to avoid overflow in exp
    clamp_val = tl.where(neg_x > 0, neg_x, 0.0)
    exp_neg_x = tl.exp(tl.minimum(clamp_val, 88.7))  # Prevent overflow
    sigmoid_x = 1.0 / (1.0 + exp_neg_x)
    
    # Compute SiLU result
    out = x * sigmoid_x
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

def pattern(input_tensor):
    """Pattern: SiLU (Swish) activation function"""
    # SiLU is defined as x * sigmoid(x) = x / (1 + exp(-x))
    result = torch.nn.functional.silu(input_tensor, inplace=True)
    return result

def replacement_args(input_tensor):
    """Extract arguments needed for replacement - just the input tensor"""
    return (input_tensor,)

def replacement_func():
    """Return the optimized kernel implementation for SiLU activation"""
    return optimized_silu