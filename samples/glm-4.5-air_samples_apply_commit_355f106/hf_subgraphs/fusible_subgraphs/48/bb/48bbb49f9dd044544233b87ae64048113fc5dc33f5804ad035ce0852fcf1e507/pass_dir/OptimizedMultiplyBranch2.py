import torch
import triton
import triton.language as tl

def pattern(reshaped_tensor, in_1):
    # Pattern matches: multiply + reshape + sum for the second computation branch
    # This is for the second computation branch: tmp_3.mul(tmp_1)
    # which gets reshaped and then summed along dim=2
    
    multiply_result = reshaped_tensor.mul(in_1)
    reshaped_for_sum = multiply_result.reshape(reshaped_tensor.shape[0], 17, -1)
    sum_result = torch.sum(reshaped_for_sum, dim=2, keepdim=True)
    
    return multiply_result, reshaped_for_sum, sum_result

def replacement_args(reshaped_tensor, in_1):
    return (reshaped_tensor, in_1)

@triton.jit
def optimized_multiply_reshape_sum_kernel(
    reshaped_ptr,    # Shape: [batch_size, 17, 64, 64]
    in_1_ptr,        # Shape: [1, 1, 64, 1]
    sum_out_ptr,     # Output: [batch_size, 17, 64, 1]
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    total_elems = batch_size * 17 * 64 * 64
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elems
    
    # Load reshaped tensor values
    reshaped_vals = tl.load(reshaped_ptr + offsets, mask=mask, other=0.0)
    
    # Broadcast in_1 - it depends on the height dimension (second last dimension)
    orig_offsets = offsets
    width = orig_offsets % 64
    height = (orig_offsets // 64) % 64
    group = (orig_offsets // (64 * 64)) % 17
    batch = orig_offsets // (64 * 64 * 17)
    
    # Load in_1 value (broadcasted across all dimensions except the second last)
    in_1_vals = tl.load(in_1_ptr + height, mask=height < 64, other=0.0)
    
    # Multiply element-wise
    multiplied = reshaped_vals * in_1_vals
    
    # Sum over the flattened dimension (dim=2 which is height * width)
    # We sum to get [batch_size, 17, 64, 1]
    # For each (batch, group), sum over height and width dimensions
    # We sum over height first, then the results will be used in a second reduction
    # For now, let's sum over height dimension
    batch_group_width_offsets = batch * 17 * 64 * 64 + group * 64 * 64 + width
    batch_group_width_mask = batch_group_width_offsets + tl.arange(0, 64) < total_elems
    
    # Load all height elements for this batch, group, and width
    height_elems = tl.load(reshaped_ptr + batch_group_width_offsets + tl.arange(0, 64) * 64, mask=batch_group_width_mask, other=0.0)
    height_elems_in_1 = tl.load(in_1_ptr + tl.arange(0, 64), mask=tl.arange(0, 64) < 64, other=0.0)
    
    # Sum over height dimension
    summed_over_height = tl.sum(height_elems * height_elems_in_1, axis=0)
    
    # Store intermediate summed result (this will need final reduction over width)
    intermediate_offset = batch * 17 * 1 * 64 + group * 1 * 64 + width
    intermediate_mask = intermediate_offset < (batch_size * 17 * 64)
    
    if intermediate_mask:
        tl.store(sum_out_ptr + intermediate_offset, summed_over_height)

@torch.fx.wrap  
def optimized_multiply_reshape_sum(reshaped_tensor, in_1):
    batch_size = reshaped_tensor.shape[0]
    
    # Create output tensor
    sum_out = torch.empty((batch_size, 17, 1 * 64), dtype=torch.float32, device=reshaped_tensor.device)
    
    # Launch kernel
    num_programs = (batch_size * 17 * 64 * 64 + 1023) // 1024
    optimized_multiply_reshape_sum_kernel[(num_programs,)](
        reshaped_tensor,
        in_1,
        sum_out,
        batch_size,
        BLOCK_SIZE=1024
    )
    
    # Note: This kernel does a partial reduction over height dimension
    # We need a second reduction step for width dimension
    final_sum = torch.sum(sum_out.reshape(batch_size, 17, 64, -1), dim=2, keepdim=True)
    
    return final_sum

def replacement_func():
    return optimized_multiply_reshape_sum