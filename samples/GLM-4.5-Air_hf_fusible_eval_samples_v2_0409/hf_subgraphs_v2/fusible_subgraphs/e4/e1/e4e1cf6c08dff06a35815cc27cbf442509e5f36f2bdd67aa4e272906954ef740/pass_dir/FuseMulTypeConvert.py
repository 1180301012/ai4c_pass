import torch
import triton
import triton.language as tl

# Pattern matching for multiplication and type conversion operations
def pattern(scaled_norm, weight_offset, original_input):
    tmp_12 = scaled_norm * weight_offset
    tmp_13 = tmp_12.type_as(original_input)
    return tmp_13

# Argument extraction function
def replacement_args(scaled_norm, weight_offset, original_input):
    return (scaled_norm, weight_offset, original_input)

@triton.jit
def fused_mul_type_convert_kernel(
    norm_ptr,
    weight_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both tensors
    norm_data = tl.load(norm_ptr + offsets, mask=mask, other=0.0)
    weight_data = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    
    # Perform multiplication and convert to output type
    # Note: Triton handles type conversion automatically based on data types
    result = norm_data * weight_data
    
    # Store result (the output type will be handled by PyTorch)
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_mul_type_convert(scaled_norm, weight_offset, original_input):
    # Ensure both tensors have the same number of elements
    assert scaled_norm.numel() == weight_offset.numel(), "Tensor size mismatch"
    
    n_elements = scaled_norm.numel()
    
    if n_elements == 0:
        return torch.zeros_like(original_input)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with target type
    output = torch.empty_like(original_input, device=scaled_norm.device)
    
    # Launch kernel
    fused_mul_type_convert_kernel[(num_programs,)](
        norm_ptr=scaled_norm,
        weight_ptr=weight_offset,
        out_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_mul_type_convert