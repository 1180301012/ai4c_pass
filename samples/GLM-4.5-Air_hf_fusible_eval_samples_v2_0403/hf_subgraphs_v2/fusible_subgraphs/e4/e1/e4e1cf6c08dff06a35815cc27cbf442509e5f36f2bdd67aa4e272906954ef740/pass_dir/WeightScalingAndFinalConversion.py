import torch
import triton
import triton.language as tl

def pattern(normalized_tensor, scalar_multiply_result, weight):
    """Match the weight scaling and final type conversion pattern"""
    # Step 1: Convert weight to float
    tmp_10 = weight.float()
    
    # Step 2: Add 1.0 to converted weight
    tmp_11 = 1.0 + tmp_10
    
    # Step 3: Multiply normalized result by scale factor
    tmp_12 = normalized_tensor * tmp_11
    
    # Step 4: Convert back to original type of scalar_multiply_result
    tmp_13 = tmp_12.type_as(scalar_multiply_result)
    
    # Return the final converted result
    return tmp_13

def replacement_args(normalized_tensor, scalar_multiply_result, weight):
    return (normalized_tensor, scalar_multiply_result, weight)

@triton.jit
def weight_scaling_kernel(
    normalized_ptr,
    scalar_multiply_ptr,
    weight_ptr,
    final_out_ptr,
    n_elements,
    n_weight_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.range(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load normalized tensor
    normalized = tl.load(normalized_ptr + offsets, mask=mask, other=0.0)
    
    # Load weight (bfloat16, needs to be converted to float)
    weight_idx = offsets % n_weight_dim
    weight = tl.load(weight_ptr + weight_idx, mask=weight_idx < n_weight_dim, other=1.0)
    
    # Convert weight to float and compute scale factor (1.0 + float(weight))
    weight_float = weight.to(tl.float32)
    scale_factor = 1.0 + weight_float
    
    # Apply scaling
    scaled_result = normalized * scale_factor
    
    # Convert back to original type (bfloat16)
    final_result = scaled_result.to(tl.bfloat16)
    
    # Store final result
    tl.store(final_out_ptr + offsets, final_result, mask=mask)

@torch.fx.wrap
def weight_scaling_operations(normalized_tensor, scalar_multiply_result, weight):
    """Weight scaling and final type conversion"""
    n_elements = normalized_tensor.numel()
    n_weight_dim = weight.numel()
    
    # Create output tensor
    final_out = torch.empty_like(scalar_multiply_result)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    weight_scaling_kernel[(num_programs,)](
        normalized_ptr=normalized_tensor,
        scalar_multiply_ptr=scalar_multiply_result,
        weight_ptr=weight,
        final_out_ptr=final_out,
        n_elements=n_elements,
        n_weight_dim=n_weight_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return final_out

def replacement_func():
    return weight_scaling_operations