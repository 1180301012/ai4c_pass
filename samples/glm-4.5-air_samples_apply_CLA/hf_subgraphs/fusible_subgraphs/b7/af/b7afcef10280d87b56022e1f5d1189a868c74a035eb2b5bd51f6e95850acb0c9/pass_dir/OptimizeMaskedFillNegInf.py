import torch
import triton
import triton.language as tl

def pattern(tensor, mask):
    # This matches the computation pattern:
    # tensor.masked_fill(mask, -3.4028234663852886e+38)
    return tensor.masked_fill(mask, -3.4028234663852886e+38)

def replacement_args(tensor, mask):
    return (tensor, mask)

@triton.jit
def optimized_masked_fill_kernel(
    tensor_ptr,
    mask_ptr,
    output_ptr,
    neg_inf_value: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load tensor and mask values
    tensor_vals = tl.load(tensor_ptr + offsets, mask=mask, other=0.0)
    mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=False)
    
    # Perform masked fill: if mask is True, use neg_inf_value, otherwise use tensor value
    # We use the ternary operator from triton.language
    result = tl.where(mask_vals, neg_inf_value, tensor_vals)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_masked_fill(tensor, mask):
    # Validate inputs
    if tensor.dtype != torch.float32:
        raise ValueError("Tensor must be float32")
    if mask.dtype != torch.bool:
        raise ValueError("Mask must be boolean")
    if tensor.shape != mask.shape:
        raise ValueError("Tensor and mask must have the same shape")
    
    n_elements = tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(tensor)
    
    # Launch kernel with negative infinity value
    neg_inf = -3.4028234663852886e+38
    optimized_masked_fill_kernel[(num_programs,)](
        tensor_ptr=tensor,
        mask_ptr=mask,
        output_ptr=output,
        neg_inf_value=neg_inf,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_masked_fill