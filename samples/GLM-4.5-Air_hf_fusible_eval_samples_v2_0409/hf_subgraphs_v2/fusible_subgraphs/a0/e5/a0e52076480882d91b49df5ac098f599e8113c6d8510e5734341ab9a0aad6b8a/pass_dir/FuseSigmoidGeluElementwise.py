import torch
import triton
import triton.language as tl

@triton.jit
def fused_sigmoid_gelu_kernel(
    conv_ptr, in_2_ptr, out_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * channels * height * width
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements
    
    # Load conv output and multiply with in_2 (element-wise)
    conv_out = tl.load(conv_ptr + offset, mask=mask, other=0.0)
    in_2_val = tl.load(in_2_ptr + offset, mask=mask, other=0.0)
    
    # Apply sigmoid * in_2 * gelu (fused)
    sigmoid_val = tl.sigmoid(conv_out)
    tmp_4 = in_2_val * sigmoid_val
    
    # GELU approximation (using polynomial approximation for better performance)
    gelu_val = tmp_4 * 0.5 * (1.0 + tl.tanh(tmp_4 * 0.7978845608 * (1.0 + 0.044715 * tmp_4 * tmp_4)))
    
    # Store result
    tl.store(out_ptr + offset, gelu_val, mask=mask)

def pattern(tmp_3, in_2):
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    return tmp_5

def replacement_args(tmp_3, in_2):
    return (tmp_3, in_2)

@torch.fx.wrap
def fused_sigmoid_gelu_forward(tmp_3, in_2):
    """Fused element-wise multiplication with gelu operation"""
    batch_size, channels, height, width = tmp_3.shape
    
    # Calculate optimal block size and grid
    total_elements = batch_size * channels * height * width
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(conv2d)
    
    # Launch fused kernel
    fused_sigmoid_gelu_kernel[(num_programs,)](
        conv2d,
        in_2,
        output,
        batch_size,
        channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_sigmoid_gelu_forward