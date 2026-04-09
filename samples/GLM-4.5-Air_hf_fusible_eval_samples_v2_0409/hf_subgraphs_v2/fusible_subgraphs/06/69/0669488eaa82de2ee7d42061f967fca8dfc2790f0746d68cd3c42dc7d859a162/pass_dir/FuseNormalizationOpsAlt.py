import torch
import triton
import triton.language as tl

# Pattern matching for normalization operations with different scaling factor
def pattern(tmp_2, tmp_3):
    """Match the normalization operations: L2 norm -> scale -> clamp -> divide"""
    tmp_4 = tmp_3 * 0.07216878364870322  # Alternative scaling factor
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    return tmp_6

# Argument extraction function
def replacement_args(tmp_2, tmp_3):
    # Extract the scaling factor from the multiplication
    scale_factor = 0.07216878364870322  # Alternative scaling factor
    return (tmp_2, tmp_3, scale_factor)

# Custom Triton kernel for fused normalization - simplified 1D approach
@triton.jit
def fused_norm_kernel(
    input_ptr,      # Pointer to flattened input tensor (tmp_2)
    norm_ptr,       # Pointer to norm values (tmp_3) 
    output_ptr,     # Pointer to output tensor
    scale_factor,   # Scaling factor for norm
    norm_stride,    # Stride for norm values
    n_elements,     # Total number of elements in input
    BLOCK_SIZE: tl.constexpr,
):
    """Fused normalization kernel with proper broadcasting using 1D grid"""
    # Each program handles a 1D block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate norm index (broadcasting: each norm value covers multiple input elements)
    # norm_stride determines how many input elements share the same norm value
    norm_indices = offsets // norm_stride
    norm_data = tl.load(norm_ptr + norm_indices, mask=mask, other=0.0)
    
    # Perform fused normalization operations
    scaled_norm = norm_data * scale_factor
    clamped_norm = tl.maximum(scaled_norm, 1e-05)
    output_data = input_data / clamped_norm
    
    # Store result
    tl.store(output_ptr + offsets, output_data, mask=mask)

@torch.fx.wrap
def fused_normalization(input_tensor, norm_tensor, scale_factor):
    """Wrapper function for fused normalization with stride-based broadcasting"""
    n_elements = input_tensor.numel()
    
    # Calculate stride: ratio of input elements to norm elements
    norm_elements = norm_tensor.numel()
    norm_stride = n_elements // norm_elements
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    output_tensor = torch.empty_like(input_tensor)
    
    # Launch Triton kernel with 1D grid
    fused_norm_kernel[(num_programs,)](
        input_ptr=input_tensor,
        norm_ptr=norm_tensor,
        output_ptr=output_tensor,
        scale_factor=scale_factor,
        norm_stride=norm_stride,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

# Replacement function (returns the optimized kernel wrapper)
def replacement_func():
    return fused_normalization