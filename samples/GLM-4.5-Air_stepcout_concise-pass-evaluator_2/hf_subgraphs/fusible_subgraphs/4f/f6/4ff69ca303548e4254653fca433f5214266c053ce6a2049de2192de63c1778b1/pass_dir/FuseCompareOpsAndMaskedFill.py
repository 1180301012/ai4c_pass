import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Pattern matches the computation exactly from model.py"""
    tmp_0 = in_0 != 0
    tmp_1 = in_0.masked_fill(tmp_0, -1000.0)
    tmp_2 = in_0 == 0
    return (tmp_2, tmp_1)

def replacement_args(in_0):
    """Extract input tensor for the replacement kernel"""
    return (in_0,)

@triton.jit
def fused_comparison_and_fill_kernel(
    input_ptr,
    bool_mask_zero_ptr,
    bool_mask_nonzero_ptr,
    filled_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    fill_value: tl.constexpr,
):
    """Fused kernel that computes both boolean masks and applies masked fill in one pass"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input value
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute boolean masks: for elements that are zero and non-zero
    is_zero = (x == 0.0)
    is_nonzero = ~is_zero
    
    # Store both boolean masks
    tl.store(bool_mask_zero_ptr + offsets, is_zero, mask=mask)
    tl.store(bool_mask_nonzero_ptr + offsets, is_nonzero, mask=mask)
    
    # Apply masked fill: replace non-zero values with fill_value
    filled_data = tl.where(is_nonzero, fill_value, x)
    tl.store(filled_ptr + offsets, filled_data, mask=mask)

@torch.fx.wrap
def fused_comparison_and_fill(in_0):
    """Wrapper function that launches the fused kernel"""
    # Get tensor properties
    n_elements = in_0.numel()
    shape = in_0.shape
    device = in_0.device
    dtype = in_0.dtype
    
    # Create output tensors
    bool_mask_zero = torch.zeros(shape, dtype=torch.bool, device=device)
    bool_mask_nonzero = torch.zeros(shape, dtype=torch.bool, device=device)
    filled_output = torch.empty_like(input_tensor)
    
    # Kernel configuration - tuned for performance
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the fused kernel
    fused_comparison_and_fill_kernel[(num_programs,)](
        input_ptr=in_0,
        bool_mask_zero_ptr=bool_mask_zero,
        bool_mask_nonzero_ptr=bool_mask_nonzero,
        filled_ptr=filled_output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        fill_value=-1000.0,
    )
    
    return bool_mask_zero, filled_output

def replacement_func():
    """Returns the optimized fused function"""
    return fused_comparison_and_fill