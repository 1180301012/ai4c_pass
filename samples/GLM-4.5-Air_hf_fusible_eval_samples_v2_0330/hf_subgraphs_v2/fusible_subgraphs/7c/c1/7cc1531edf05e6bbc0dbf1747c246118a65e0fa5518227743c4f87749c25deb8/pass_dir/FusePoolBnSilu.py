import torch
import triton
import triton.language as tl

def pattern(in_4):
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    return tmp_4

def replacement_args(in_4):
    return (in_4,)

@triton.jit
def optimized_reshape_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Simple reshape: [4, 128, 256] -> [1, 512, 16, 16]
    # Direct copy since total elements are the same: 4*128*256 = 131072
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Store result (no transformation needed for reshape)
    tl.store(output_ptr + offsets, input_val, mask=mask)

@torch.fx.wrap
def optimized_reshape(input_tensor):
    # Output shape after reshape: [1, 512, 16, 16]
    output_shape = [1, 512, 16, 16]
    output_size = input_tensor.numel()  # Same number of elements, just reshaped
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Triton kernel launch configuration with better performance
    BLOCK_SIZE = 2048  # Larger block size for better GPU utilization
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized kernel
    optimized_reshape_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=output_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_reshape