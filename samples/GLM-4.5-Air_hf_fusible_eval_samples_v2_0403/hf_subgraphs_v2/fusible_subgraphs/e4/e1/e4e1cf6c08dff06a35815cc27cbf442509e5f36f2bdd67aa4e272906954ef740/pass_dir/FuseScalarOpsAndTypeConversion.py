import torch
import triton
import triton.language as tl

def pattern(x, scalar, weight):
    """Match the scalar operations and type conversion pattern"""
    # Step 1: Multiply input with scalar (already bfloat16 to float)  
    tmp_4 = x.float()
    
    # Step 2: Convert weight to float and add 1.0
    tmp_11 = 1.0 + weight.float()
    
    # Step 3: Multiply normalized result (from elsewhere) with scale factor
    # We'll get the normalized result from the pattern that calls this
    tmp_12 = tmp_4 * tmp_11
    
    # Step 4: Convert back to original type
    tmp_13 = tmp_12.type_as(x)
    
    # Return intermediate float result and final converted result
    return tmp_4, tmp_13

def replacement_args(x, scalar, weight):
    return (x, scalar, weight)

@triton.jit
def fused_scalar_ops_kernel(
    x_in_ptr,
    scalar_ptr,
    weight_ptr,
    float_x_out_ptr,
    final_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x_in = tl.load(x_in_ptr + offsets, mask=mask, other=0.0)
    scalar = tl.load(scalar_ptr) if scalar_ptr is not None else 1.0
    weight = tl.load(weight_ptr + (offsets % 2048), mask=offsets % 2048 < 2048, other=1.0)
    
    # Convert float
    x_float = x_in.to(tl.float32)
    
    # Apply scalar multiplication and scale factor
    # Apply scalar multiplication (this assumes x is already scaled from previous operation)
    scaled_x = x_float * scalar
    
    # Apply weight scaling (1.0 + float(weight))
    weight_float = weight.to(tl.float32)
    scale_factor = 1.0 + weight_float
    
    # Final scaling
    final_float = scaled_x * scale_factor
    
    # Convert back to original type (bfloat16)
    final_result = final_float.to(tl.bfloat16)
    
    # Store both intermediate float and final result
    tl.store(float_x_out_ptr + offsets, x_float, mask=mask)
    tl.store(final_out_ptr + offsets, final_result, mask=mask)

@torch.fx.wrap
def fused_scalar_operations(x, scalar, weight):
    """Fused scalar operations with type conversions"""
    n_elements = x.numel()
    
    # Create output tensors
    float_x_out = torch.empty(x.shape, dtype=torch.float32, device='cuda')
    final_out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_scalar_ops_kernel[(num_programs,)](
        x_in_ptr=x,
        scalar_ptr=scalar,
        weight_ptr=weight,
        float_x_out_ptr=float_x_out,
        final_out_ptr=final_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return float_x_out, final_out

def replacement_func():
    return fused_scalar_operations