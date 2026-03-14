import torch
import triton
import triton.language as tl

# Pattern matching function - matches the simple masked fill operation
def pattern(in_0):
    """
    Matches the masked fill operation:
    tmp_0 = in_0 != 0
    tmp_1 = in_0.masked_fill(tmp_0, -1000.0)
    Returns tmp_1
    """
    tmp_0 = in_0 != 0
    tmp_1 = in_0.masked_fill(tmp_0, -1000.0)
    return tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel using Triton
@triton.jit
def mask_operations_kernel(
    input_ptr,
    zero_mask_out_ptr,
    filled_out_ptr,
    n_elements,
    fill_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that:
    1. Creates zero mask (input == 0)
    2. Creates filled tensor (non-zero elements replaced with fill_value)
    Both operations are done in a single kernel pass for better cache efficiency
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Create zero mask (input == 0) - using small epsilon for float comparisons
    zero_mask = (tl.abs(input_vals) < 1e-6)
    
    # Create filled tensor
    # For non-zero elements, apply fill_value; keep original for zero elements
    filled_vals = tl.where(zero_mask, input_vals, fill_value)
    
    # Store both results
    tl.store(zero_mask_out_ptr + offsets, zero_mask, mask=mask)
    tl.store(filled_out_ptr + offsets, filled_vals, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_mask_operations(input_tensor, fill_value=-1000.0):
    """
    Wrapper function that launches the optimized kernel
    Returns only the filled tensor
    """
    N = input_tensor.numel()
    
    # Optimized kernel for just the masked fill operation
    @triton.jit
    def masked_fill_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        fill_value: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Optimized kernel for masked fill operation"""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load input data
        input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # Create optimized masked fill operation
        # Use exact zero comparison like PyTorch's == 0 for floats
        zero_mask = (input_vals == 0.0)
        output_vals = tl.where(zero_mask, input_vals, fill_value)
        
        # Store result
        tl.store(output_ptr + offsets, output_vals, mask=mask)
    
    # Optimize block size based on tensor size
    # Tensor: [1, 361, 49, 49] = 866,281 elements
    # Use medium block size for better GPU utilization and memory access pattern
    BLOCK_SIZE = 1024  # Balanced block size for this tensor size
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    masked_fill_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=N,
        fill_value=fill_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_mask_operations