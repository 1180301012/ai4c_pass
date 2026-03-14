import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias, scale_input):
    """
    Pattern matches: conv2d -> add(1.0) -> div(2.0) -> clamp(0.0, 1.0) -> elementwise multiply
    """
    # Conv2D operation
    conv_out = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Mathematical operations that can be fused
    added = conv_out + 1.0
    divided = added / 2.0
    clamped = divided.clamp_(0.0, 1.0)
    
    # Final multiplication with scale factor
    multiplied = scale_input * clamped
    
    return conv_out, multiplied, scale_input

def replacement_args(conv_input, conv_weight, conv_bias, scale_input):
    return (conv_input, conv_weight, conv_bias, scale_input)

@triton.jit
def fused_math_kernel(
    conv_out_ptr,
    scale_input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data from both tensors
    conv_out = tl.load(conv_out_ptr + offsets, mask=mask, other=0.0)
    scale_input_val = tl.load(scale_input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply fused mathematical operations
    # (x + 1.0) / 2.0 -> clamp(0.0, 1.0) -> multiply by scale factor
    intermediate = (conv_out + 1.0) / 2.0
    clamped = tl.maximum(intermediate, 0.0)
    clamped = tl.minimum(clamped, 1.0)
    result = clamped * scale_input_val
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_math_operations(conv_input, conv_weight, conv_bias, scale_input):
    # Execute conv2d first using PyTorch for correctness
    conv_out = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Determine the total number of elements for parallel processing
    n_elements = conv_out.numel()
    
    # Configure block size for GPU parallel processing
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Prepare output tensor to store fused operations result
    output = torch.empty_like(conv_out)
    
    # Execute the fused mathematical operations using Triton kernel
    fused_math_kernel[(num_programs,)](
        conv_out,
        scale_input,
        output,
        n_elements,
        BLOCK_SIZE
    )
    
    return conv_out, output, scale_input

def replacement_func():
    return fused_math_operations