import torch
import triton
import triton.language as tl

@triton.jit
def optimized_mask_kernel(
    input_ptr,
    output_ptr,
    input_shape_ptr,
    mask_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < input_shape_ptr[1]  # Use sequence length from [1, 15]
    
    # Load input
    input_vals = tl.load(input_ptr + idx, mask=mask, other=0)
    
    # Generate mask: (input == 1) -> float32 -> * mask_scalar
    bool_mask = (input_vals == 1)
    float_mask = tl.cast(bool_mask, tl.float32)
    result = float_mask * mask_scalar
    
    # Store result
    tl.store(output_ptr + idx, result, mask=mask)

@torch.fx.wrap
def optimize_padding_mask_generation(input_tensor, seq_len, mask_scalar=-3.4028234663852886e+38):
    num_elements = input_tensor.numel()
    
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor, dtype=torch.float32)
    
    optimized_mask_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        input_shape_ptr=seq_len,
        mask_scalar=mask_scalar,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply unsqueeze operations
    output = output.unsqueeze(1)
    output = output.unsqueeze(1)
    
    return output

def pattern(input_tensor):
    tmp_5 = input_tensor.__eq__(1)
    tmp_6 = tmp_5.to(torch.float32)
    tmp_6 *= -3.4028234663852886e+38
    tmp_8 = tmp_6.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(1)
    return tmp_9

def replacement_args(input_tensor):
    seq_len = input_tensor.shape[1]
    return (input_tensor, seq_len)

def replacement_func():
    return optimize_padding_mask_generation