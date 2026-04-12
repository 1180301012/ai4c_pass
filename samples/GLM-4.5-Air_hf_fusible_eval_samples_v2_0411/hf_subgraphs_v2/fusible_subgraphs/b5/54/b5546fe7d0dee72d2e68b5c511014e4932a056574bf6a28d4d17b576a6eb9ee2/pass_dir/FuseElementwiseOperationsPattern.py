import torch
import triton
import triton.language as tl

def pattern(in_6, in_5, in_4, scalar_const):
    """Pattern: Element-wise multiply + pad + scalar multiply + add"""
    tmp_6 = in_6 * in_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = scalar_const * in_4
    tmp_9 = tmp_8 + tmp_7
    return tmp_9

def replacement_args(in_6, in_5, in_4, scalar_const):
    return (in_6, in_5, in_4, scalar_const)

@triton.jit
def fused_elementwise_kernel(
    input1_ptr, input2_ptr, input3_ptr, out_ptr,
    scalar_value,
    h1, w1, h3, w3,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for: multiply + pad + scalar multiply + add
    Pattern: (input1 * input2) padded + (scalar * input3)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (h1 * w1)
    
    # Load input tensors (assuming spatial dimensions)
    input1 = tl.load(input1_ptr + offsets, mask=mask, other=0.0)
    input2 = tl.load(input2_ptr + offsets, mask=mask, other=0.0)
    
    # Element-wise multiplication
    mul_result = input1 * input2
    
    # Apply padding logic: pad with 1 row at the bottom
    # For padded result, we need to handle the extra row
    padded_mask = (offsets // w1) < h1  # Only process original height elements first
    
    # Load scalar multiplied input3 
    scalar_mul = scalar_value * tl.load(input3_ptr + offsets, mask=mask, other=0.0)
    
    # Add the results
    # The padding adds zeros, so we just need to handle dimension differences
    if h3 > h1:
        # Has padding - create extended buffer for padded result
        padded_result = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
        tl.store(padded_result + offsets, mul_result, mask=padded_mask)
        
        # For padded case, extend the add operation to handle larger dimensions
        extended_mask = offsets < (h3 * w3)
        output = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
        
        # Store padded result (with zeros in padded regions)
        tl.store(output + offsets, padded_result, mask=extended_mask)
        tl.store(output + offsets, scalar_mul, mask=extended_mask & ~padded_mask)
    else:
        # No padding needed, simple addition
        output = mul_result + scalar_mul
    
    # Store final result
    tl.store(out_ptr, output, mask=mask if h3 == h1 else (offsets < (h3 * w3)))

@torch.fx.wrap
def fused_elementwise_operations(input1, input2, input3, scalar_value):
    """
    Fused operation: multiply + pad + scalar multiply + add
    Pattern: (input1 * input2) padded + (scalar * input3)
    """
    # Get spatial dimensions (assuming NCHW format, focus on HW)
    h1, w1 = input2.shape[-2:]  # input2 is the main tensor for spatial dims
    h3, w3 = input3.shape[-2:]
    
    # Create output tensor
    output_shape = input3.shape
    output = torch.empty(output_shape, dtype=input3.dtype, device=input3.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    total_elements = h1 * w1
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_elementwise_kernel[(num_programs,)](
        input1, input2, input3, output,
        scalar_value,
        h1, w1, h3, w3,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_elementwise_operations