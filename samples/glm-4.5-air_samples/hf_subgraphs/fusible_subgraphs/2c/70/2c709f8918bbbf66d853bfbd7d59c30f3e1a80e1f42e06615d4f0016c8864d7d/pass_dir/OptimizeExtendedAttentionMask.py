import torch
import triton
import triton.language as tl

def pattern(extended_attention_mask):
    # Convert int64 to float32
    tmp_1 = extended_attention_mask.to(dtype=torch.float32)
    # Apply mask transformation: 1.0 - x * -3.4e38
    tmp_2 = 1.0 - tmp_1
    tmp_3 = tmp_2 * -3.4028234663852886e+38
    # Return the final mask result
    return tmp_3

def replacement_args(extended_attention_mask):
    return (extended_attention_mask,)

@triton.jit
def extended_attention_mask_kernel(
    mask_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load values
    mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=0)
    
    # Direct arithmetic operation in one step for maximum efficiency
    # This avoids any intermediate variables and computations
    result = (-3.4028234663852886e+38) * (1.0 - tl.cast(mask_vals, tl.float32))
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)



@torch.fx.wrap
def optimized_extended_attention_mask(extended_attention_mask):
    # Calculate total number of elements
    num_elements = extended_attention_mask.numel()
    
    # Choose optimal block size based on tensor size for better GPU utilization
    if num_elements < 1024:
        BLOCK_SIZE = 512
    elif num_elements < 8192:
        BLOCK_SIZE = 1024
    elif num_elements < 65536:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096
    
    # Calculate grid size
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(extended_attention_mask, dtype=torch.float32)
    
    # Launch the kernel
    extended_attention_mask_kernel[(num_programs,)](
        mask_ptr=extended_attention_mask,
        output_ptr=output,
        num_elements=num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_extended_attention_mask