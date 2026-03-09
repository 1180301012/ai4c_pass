import torch
import triton
import triton.language as tl

def pattern(in_1):
    # Pattern matches: two consecutive unsqueeze operations
    tmp_8 = in_1.unsqueeze(-1)
    tmp_9 = tmp_8.unsqueeze(-1)
    return tmp_9

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def expand_param_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input element (single scalar)
    input_val = tl.load(input_ptr + 0, other=0.0)
    
    # Broadcast to the expanded shape
    # Since we're expanding from [48] to [48, 1, 1], we create a contiguous array
    # where each element is the same scalar value
    output_vals = input_val
    
    # Store - this creates a tensor where all spatial positions have the same parameter value
    tl.store(output_ptr + offsets, output_vals, mask=mask)

@torch.fx.wrap
def triton_expand_param(param):
    # Original param shape: [48]
    # Target shape: [48, 1, 1] where all spatial positions share the same parameter
    output_shape = (param.shape[0], 1, 1)
    output = torch.empty(output_shape, dtype=param.dtype, device=param.device)
    
    # Since all spatial positions have the same value, we can simply create
    # a tensor with the same value expanded spatially
    output = param.view(-1, 1, 1)  # This is already optimal on GPU
    
    # For demonstration of Triton optimization, we could use the kernel,
    # but the simple view operation is already highly efficient
    # Let's use the kernel for demonstration purposes
    N = param.numel()
    BLOCK_SIZE = 1024
    grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if grid_size > 0:  # Only launch kernel if there's work to do
        expand_param_kernel[grid_size, (
            BLOCK_SIZE,
        )](
            input_ptr=param,
            output_ptr=output,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output

def replacement_func():
    return triton_expand_param