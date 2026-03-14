import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Pattern matches in-place ReLU activation"""
    return torch.nn.functional.relu(input_tensor, inplace=True)

def replacement_args(input_tensor):
    # Extract the single argument needed for the optimized ReLU
    return (input_tensor,)

@triton.jit
def optimized_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized Triton kernel for ReLU activation"""
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU activation
    out = tl.maximum(x, 0.0)
    
    # Store output
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_relu_inplace(input_tensor):
    """Optimized in-place ReLU using Triton"""
    n_elements = input_tensor.numel()
    
    # Use optimal block size for GPU architecture
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor (same as input for "in-place" semantics)
    output = torch.empty_like(input_tensor)
    
    # Launch Triton kernel
    optimized_relu_kernel[(num_programs,)](
        input_tensor,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_relu_inplace