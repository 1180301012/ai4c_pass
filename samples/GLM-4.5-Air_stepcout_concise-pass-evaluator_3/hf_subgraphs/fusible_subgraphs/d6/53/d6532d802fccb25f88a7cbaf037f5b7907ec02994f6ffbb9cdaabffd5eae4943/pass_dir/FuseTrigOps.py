import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Match the trigonometric operations pattern
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    return (tmp_6, tmp_7, in_2)  # Return in_2 as well for the full pattern match

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_trigonometric_kernel(
    x_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute both cos and sin simultaneously  
    cos_val = tl.cos(x)
    sin_val = tl.sin(x)
    
    # Store both results directly as bfloat16
    tl.store(cos_out_ptr + offsets, cos_val.to(tl.bfloat16), mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val.to(tl.bfloat16), mask=mask)

@torch.fx.wrap
def fused_trigonometric_ops(in_1):
    # Determine input shape after concatenation
    original_shape = in_1.shape
    concated_shape = (*original_shape[:-1], original_shape[-1] * 2)
    
    # Ensure contiguous memory access
    if in_1.stride() != tuple(range(len(in_1.shape)-1, -1, -1)):
        in_1 = in_1.contiguous()
    
    # Create output tensors
    cos_output = torch.empty(concated_shape, dtype=torch.bfloat16, device=in_1.device)
    sin_output = torch.empty(concated_shape, dtype=torch.bfloat16, device=in_1.device)
    
    # Launch kernel
    n_elements = concated_shape.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_trigonometric_kernel[(num_programs,)](
        in_1.flatten(),
        cos_output.flatten(),
        sin_output.flatten(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return cos_output, sin_output

def replacement_func():
    def fused_trig_only(in_0, in_1, in_2):
        # Only optimize trig operations, leave rest to original
        cos_output, sin_output = fused_trigonometric_ops(in_1)
        return (cos_output, in_2, sin_output)
    
    return fused_trig_only