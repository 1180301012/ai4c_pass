import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation sequence
def pattern(in_0):
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2

# Argument extraction function - extract the input tensor
def replacement_args(in_0):
    return (in_0,)

# Triton kernel for fused multiply with transpose (softmax handled separately)
@triton.jit
def fused_multiply_kernel(
    input_ptr,
    output_ptr,
    const_val,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    program_id = tl.program_id(0)
    start_idx = program_id * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to ensure we don't go out of bounds
    mask = offsets < total_elements
    
    # Load input data with improved memory access pattern
    input_data = tl.load(
        input_ptr + offsets,
        mask=mask,
        other=0.0
    )
    
    # Apply element-wise multiplication with constant
    multiplied_data = input_data * const_val
    
    # Store the entire block result
    tl.store(
        output_ptr + offsets,
        multiplied_data,
        mask=mask
    )

# Optimized kernel wrapper
@torch.fx.wrap
def fused_multiply_softmax_transpose(input_tensor):
    n, c, h, w = input_tensor.shape
    const_val = 0.1767766952966369
    
    # Step 1: Use Triton to multiply by constant (output shape: [n, c, h, w])
    multiplied = torch.empty((n, c, h, w), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate total number of elements
    total_elements = n * c * h * w
    
    # Choose block size for better GPU utilization  
    BLOCK_SIZE = 512  # Smaller block size for better occupancy
    
    # Calculate 1D grid size with block processing
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_programs,)
    
    # Launch the kernel
    fused_multiply_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=multiplied,
        const_val=const_val,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 2: Apply softmax along last dimension (dim=-1) using PyTorch
    softmax_result = multiplied.softmax(dim=-1)
    
    # Step 3: Apply transpose of last two dimensions using PyTorch
    result = softmax_result.transpose(-2, -1)
    
    return result

# Replacement function - returns the optimized kernel function
def replacement_func():
    return fused_multiply_softmax_transpose