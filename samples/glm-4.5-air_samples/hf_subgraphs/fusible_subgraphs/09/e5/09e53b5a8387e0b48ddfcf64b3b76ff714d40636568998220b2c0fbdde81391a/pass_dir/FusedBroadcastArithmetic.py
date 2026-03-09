import torch
import triton
import triton.language as tl

def pattern(tmp_1, tmp_0, in_3):
    """
    Pattern: tmp_2 = tmp_1 * tmp_0, then tmp_3 = tmp_2 + in_3, then tmp_4 = tmp_3.contiguous()
    Matches: ((in_2 + in_1) * in_0) + in_3 and then make contiguous
    """
    tmp_2 = tmp_1 * tmp_0
    tmp_3 = tmp_2 + in_3
    tmp_4 = tmp_3.contiguous()
    return tmp_4

def replacement_args(tmp_1, tmp_0, in_3):
    """
    Extract arguments for the fused kernel:
    - tmp_1: the updated in_2 (after in_2 += in_1)  
    - tmp_0: the scalar in_0
    - in_3: the third input tensor
    """
    return (tmp_1, tmp_0, in_3)

@triton.jit
def fused_arithmetic_kernel(
    x_ptr,      # tmp_1 pointer
    y_value,    # tmp_0 value (scalar) passed as Python scalar
    z_ptr,      # in_3 pointer
    out_ptr,    # output pointer
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: ((x * y) + z)
    Where:
    - x is tmp_1 (the updated in_2 after in_2 += in_1)  
    - y is tmp_0 (the scalar in_0)
    - z is in_3
    """
    # Each program handles a contiguous block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Ensure we don't go out of bounds
    
    # Load x tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load z tensor
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: ((x * y_value) + z)
    # This is equivalent to: ((in_2 + in_1) * in_0) + in_3
    # y_value is automatically broadcast to all elements by Triton
    result = (x * y_value) + z
    
    # Store directly to output (contiguous by design)
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_arithmetic_wrapper(tmp_1, tmp_0, in_3):
    """
    Wrapper function to launch the fused kernel
    """
    # All inputs should be the same shape
    assert tmp_1.shape == in_3.shape, "Input tensors must have the same shape"
    
    total_elements = tmp_1.numel()
    output_shape = tmp_1.shape
    
    # Create output tensor with same properties as input
    out = torch.empty(output_shape, dtype=tmp_1.dtype, device=tmp_1.device)
    
    # Use optimal block size that balances performance across tensor sizes
    BLOCK_SIZE = 256
    
    # Calculate grid size
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Extract scalar value from tmp_0 (it's a shape [] tensor)
    scalar_value = tmp_0.item()
    
    # Launch kernel with scalar value directly
    fused_arithmetic_kernel[(num_programs,)](
        x_ptr=tmp_1,
        y_value=scalar_value,  # Pass scalar value directly
        z_ptr=in_3,
        out_ptr=out,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """
    Return the fused arithmetic function as the replacement
    """
    return fused_arithmetic_wrapper