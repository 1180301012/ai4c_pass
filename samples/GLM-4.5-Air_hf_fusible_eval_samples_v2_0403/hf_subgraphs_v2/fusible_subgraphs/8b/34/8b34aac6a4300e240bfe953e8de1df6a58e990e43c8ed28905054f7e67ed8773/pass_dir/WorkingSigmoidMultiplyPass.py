import torch
import triton
import triton.language as tl

def pattern(conv_output, multiply_input):
    """
    Pattern to match sigmoid followed by multiplication: sigmoid(x) * y
    This matches the pattern used in both branches of the computation
    """
    # Sigmoid activation followed by multiplication
    sigmoid_result = torch.sigmoid(conv_output)
    final_result = sigmoid_result * multiply_input
    return final_result

def replacement_args(conv_output, multiply_input):
    return (conv_output, multiply_input)

@triton.jit
def sigmoid_multiply_kernel(
    conv_output_ptr, multiply_input_ptr, output_ptr,
    n_elements: tl.constexpr,
):
    """
    Optimized Triton kernel for fused sigmoid and multiplication
    """
    pid = tl.program_id(0)
    block_start = pid * 1024
    offsets = block_start + tl.arange(0, 1024)
    mask = offsets < n_elements
    
    # Load inputs
    conv_output = tl.load(conv_output_ptr + offsets, mask=mask, other=0.0)
    multiply_input = tl.load(multiply_input_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: sigmoid(conv_output) * multiply_input
    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_output))
    result = sigmoid_val * multiply_input
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_sigmoid_multiply(conv_output, multiply_input):
    """
    Wrapper function for fused sigmoid and multiplication
    """
    # Flatten inputs for efficient processing
    conv_flat = conv_output.flatten()
    multiply_flat = multiply_input.flatten()
    
    n_elements = conv_flat.numel()
    output = torch.empty_like(conv_flat)
    
    # Launch kernel
    grid_size = (n_elements + 1023) // 1024
    sigmoid_multiply_kernel[grid_size](
        conv_flat, multiply_flat, output,
        n_elements
    )
    
    # Return in original shape
    return output.reshape(conv_output.shape)

def replacement_func():
    return fused_sigmoid_multiply