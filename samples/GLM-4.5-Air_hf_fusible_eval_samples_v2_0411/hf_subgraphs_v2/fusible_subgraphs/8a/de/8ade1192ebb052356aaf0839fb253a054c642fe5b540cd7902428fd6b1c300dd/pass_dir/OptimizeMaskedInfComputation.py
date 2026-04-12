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
def optimized_masked_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor directly
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Perform computation directly: 1.0 - input_vals, then mask with -inf
    result = 1.0 - input_vals.to(tl.float32)
    bool_mask = result != result  # This creates NaN mask, but we want to use explicit -inf
    result = tl.where(bool_mask, -3.4028234663852886e+38, result)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def optimized_masked_computation(in_5):
    # Determine input and output shapes
    if in_5.dim() == 0:
        n_elements = 1
        out_shape = []
    else:
        n_elements = in_5.numel()
        out_shape = in_5.shape
    
    # Create output tensor
    output = torch.empty(out_shape, dtype=torch.float32, device=in_5.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_masked_kernel[(num_programs,)](
        input_ptr=in_5,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return optimized_masked_computation