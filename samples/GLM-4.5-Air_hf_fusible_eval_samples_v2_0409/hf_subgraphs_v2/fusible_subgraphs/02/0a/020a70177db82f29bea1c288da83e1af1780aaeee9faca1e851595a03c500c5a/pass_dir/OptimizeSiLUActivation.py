import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Pattern: SiLU activation optimization"""
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    return tmp_0

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def silu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized SiLU kernel: x * sigmoid(x)"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU: x * sigmoid(x)
    # Use fast sigmoid approximation for better performance
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_out = x * sigmoid_x
    
    # Store result
    tl.store(output_ptr + offsets, silu_out, mask=mask)

@torch.fx.wrap
def optimized_silu(input_tensor):
    """Optimized SiLU activation using Triton"""
    n_elements = input_tensor.numel()
    
    # Use block size optimized for GPU architecture
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    silu_kernel[(num_programs, 1, 1)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_silu