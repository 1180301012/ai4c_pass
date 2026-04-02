import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    """Match Conv2D → Hardswish (inplace) → Flatten pattern"""
    conv2d = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    """Extract arguments for the replacement kernel"""
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def simple_fused_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements: tl.constexpr,
):
    """Simple fused kernel - one element per thread"""
    pid = tl.program_id(0)
    
    if pid >= n_elements:
        return
    
    # For now, just copy input to output to test shape handling
    # This will help us debug the shape issue first
    # input_val = tl.load(input_ptr + pid)
    # tl.store(output_ptr + pid, input_val)
    
    # Just store zeros to test output allocation
    tl.store(output_ptr + pid, 0.0)

@torch.fx.wrap
def simple_fused_conv_hardswish_flatten(input_tensor, weight_tensor, bias_tensor):
    """
    Simple fused operation to test shape handling
    """
    # Get output shape by computing the actual operation first (simple approach)
    conv_output = torch.conv2d(input_tensor, weight_tensor, bias_tensor, 
                              stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    flattened_output = conv_output.flatten(1, -1)
    
    # Allocate output tensor with the correct shape
    output = torch.empty_like(flattened_output)
    
    # Get total number of elements
    n_elements = output.numel()
    
    # Use simple block size
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch simple kernel
    simple_fused_kernel[(grid_size,)](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output,
        n_elements=BLOCK_SIZE,  # Using block size for now
    )
    
    return output

def replacement_func():
    """Returns the fused kernel function"""
    return simple_fused_conv_hardswish_flatten