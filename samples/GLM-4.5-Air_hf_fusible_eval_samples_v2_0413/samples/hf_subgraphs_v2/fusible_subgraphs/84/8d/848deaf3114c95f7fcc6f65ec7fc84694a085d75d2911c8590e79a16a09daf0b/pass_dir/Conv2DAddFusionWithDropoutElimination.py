import torch
import triton
import triton.language as tl

@triton.jit
def elementwise_add_kernel(
    x_ptr,
    y_ptr, 
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise addition kernel for Triton"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

def pattern(conv_input, weight, bias, add_input):
    # Pattern matches the original computation:
    # conv2d = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    # dropout_result = torch.nn.functional.dropout(conv2d, 0.0, False, False)  # p=0.0 = identity
    # final_result = dropout_result + add_input
    # Return all observable values (conv2d_result, dropout_result, final_result)
    
    conv2d_result = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    dropout_result = torch.nn.functional.dropout(conv2d_result, 0.0, False, False)
    final_result = dropout_result + add_input
    
    return conv2d_result, dropout_result, final_result

def replacement_args(conv_input, weight, bias, add_input):
    return (conv_input, weight, bias, add_input)

@torch.fx.wrap 
def optimized_conv2d_add_fusion(conv_input, weight, bias, add_input):
    """
    Optimized implementation that eliminates dropout (p=0.0) and uses Triton for elementwise addition
    """
    # Step 1: Compute conv2d (dropout with p=0.0 is identity, so we skip it)
    conv_result = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Step 2: Element-wise addition using Triton kernel
    n_elements = conv_result.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(conv_result)
    
    # Launch Triton kernel for element-wise addition
    elementwise_add_kernel[(num_programs,)](
        conv_result,
        add_input,
        output,
        n_elements,
        BLOCK_SIZE
    )
    
    # Return all observable values: conv_result, "dropout_result" (same as conv), final result
    return conv_result, conv_result, output

def replacement_func():
    return optimized_conv2d_add_fusion