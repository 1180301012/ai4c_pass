import torch
import triton
import triton.language as tl

def pattern(conv_output, input_feat):
    # Residual connection pattern
    tmp_5 = conv_output + input_feat
    return tmp_5

def replacement_args(conv_output, input_feat):
    return (conv_output, input_feat)



@torch.fx.wrap
def depthwise_conv_residual(conv_output, input_feat):
    batch_size, channels, height, width = input_feat.shape
    
    output = torch.empty_like(input_feat)
    
    # Simple residual addition (since we're only matching the addition part)
    # This is just: output = conv_output + input_feat
    # We'll implement this efficiently with Triton
    total_elements = batch_size * channels * height * width
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    residual_add_kernel[(num_programs,)](
        conv_output_ptr=conv_output,
        input_feat_ptr=input_feat,
        output_ptr=output,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@triton.jit
def residual_add_kernel(
    conv_output_ptr, input_feat_ptr, output_ptr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid * BLOCK_SIZE >= total_elements:
        return
        
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    conv_val = tl.load(conv_output_ptr + offsets, mask=mask, other=0.0)
    input_val = tl.load(input_feat_ptr + offsets, mask=mask, other=0.0)
    
    result = conv_val + input_val
    tl.store(output_ptr + offsets, result, mask=mask)

def replacement_func():
    return depthwise_conv_residual