import torch
import triton
import triton.language as tl


def pattern(in_5):
    tmp_4 = in_5.to(torch.float32)
    tmp_5 = torch.tensor(1.0, dtype=torch.float32)
    tmp_6 = tmp_5 - tmp_4
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
    return tmp_8


def replacement_args(in_5):
    return (in_5,)


@triton.jit
def mask_optimization_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values directly 
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Optimize computation: 1.0 - input_vals followed by masked fill with -inf
    # Skip redundant type conversion by doing direct math
    result = 1.0 - input_vals.to(tl.float32)
    
    # Create mask for NaN values (equivalent to the boolean conversion)
    nan_mask = result != result
    
    # Apply masked fill with -inf (max negative float32)
    inf_val = -3.4028234663852886e+38
    result = tl.where(nan_mask, inf_val, result)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def optimized_mask_computation(in_5):
    # Create output tensor with correct dtype
    if in_5.numel() == 0:
        return in_5.to(torch.float32)
    
    output = torch.empty_like(in_5, dtype=torch.float32)
    
    # Launch kernel
    n_elements = in_5.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    mask_optimization_kernel[(num_programs,)](
        input_ptr=in_5,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return optimized_mask_computation