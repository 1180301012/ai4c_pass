import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Match the pattern: bool tensor conversion"""
    # Match only the bool conversion for simplicity and reliability
    tmp_2 = in_0.to(dtype=torch.bool)
    return tmp_2

def replacement_args(in_0):
    """Extract arguments for the replacement function"""
    return (in_0,)

@triton.jit
def optimized_bool_conversion_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for boolean conversion"""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and convert to boolean efficiently
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0)
    bool_val = input_val != 0  # Efficient boolean conversion
    tl.store(output_ptr + offsets, bool_val, mask=mask)

@torch.fx.wrap  
def optimized_bool_conversion(input_tensor):
    """Optimized boolean conversion using Triton"""
    n_elements = input_tensor.numel()
    output_tensor = torch.empty(input_tensor.shape, dtype=torch.bool, device=input_tensor.device)
    
    if n_elements == 0:
        return output_tensor
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_bool_conversion_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@torch.fx.wrap
def optimized_bool_conversion_only(input_tensor):
    """Optimized implementation for bool tensor conversion"""
    return optimized_bool_conversion(input_tensor)

def replacement_func():
    """Return the optimized function"""
    return optimized_bool_conversion_only

@torch.fx.wrap
def optimized_arange(end_value, input_device):
    """Optimized arange generation using Triton"""
    output_shape = (end_value,)
    output = torch.empty(output_shape, dtype=torch.int32, device=input_device)
    if end_value == 0:
        return output
    
    BLOCK_SIZE = 1024
    num_programs = (end_value + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_arange_kernel[(num_programs,)](
        output_ptr=output,
        end_value=end_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@triton.jit
def optimized_bool_conversion_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for boolean conversion"""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and convert to boolean efficiently
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0)
    bool_val = input_val != 0  # Efficient boolean conversion
    tl.store(output_ptr + offsets, bool_val, mask=mask)

@torch.fx.wrap  
def optimized_bool_conversion(input_tensor):
    """Optimized boolean conversion using Triton"""
    n_elements = input_tensor.numel()
    output_tensor = torch.empty(input_tensor.shape, dtype=torch.bool, device=input_tensor.device)
    
    if n_elements == 0:
        return output_tensor
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_bool_conversion_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

@triton.jit
def fusion_kernel(
    arange_output_ptr,
    bool_input_ptr,
    bool_output_ptr,
    arange_end_value,
    bool_n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that handles both arange and bool conversion"""
    # Handle arange
    arange_offsets = tl.arange(0, BLOCK_SIZE)
    arange_mask = arange_offsets < arange_end_value
    tl.store(arange_output_ptr + arange_offsets, arange_offsets.to(tl.int32), mask=arange_mask)
    
    # Handle bool conversion
    bool_offsets = tl.arange(0, BLOCK_SIZE)
    bool_mask = bool_offsets < bool_n_elements
    input_val = tl.load(bool_input_ptr + bool_offsets, mask=bool_mask, other=0)
    bool_val = input_val != 0
    tl.store(bool_output_ptr + bool_offsets, bool_val, mask=bool_mask)

@torch.fx.wrap
def optimized_bool_conversion_only(input_tensor):
    """Optimized implementation for bool tensor conversion"""
    return optimized_bool_conversion(input_tensor)

@torch.fx.wrap
def empty_arange_on_device(end_value, device):
    """Create arange tensor using optimized Triton kernel"""
    output_shape = (end_value,)
    output = torch.empty(output_shape, dtype=torch.int32, device=device)
    if end_value == 0:
        return output
    
    BLOCK_SIZE = 1024
    num_programs = (end_value + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_arange_kernel[(num_programs,)](
        output_ptr=output,
        end_value=end_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@torch.fx.wrap
def optimized_bool_conversion(input_tensor):
    """Optimized boolean conversion using Triton kernel"""
    n_elements = input_tensor.numel()
    output_tensor = torch.empty(input_tensor.shape, dtype=torch.bool, device=input_tensor.device)
    
    if n_elements == 0:
        return output_tensor
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_bool_conversion_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return optimized_bool_conversion_only