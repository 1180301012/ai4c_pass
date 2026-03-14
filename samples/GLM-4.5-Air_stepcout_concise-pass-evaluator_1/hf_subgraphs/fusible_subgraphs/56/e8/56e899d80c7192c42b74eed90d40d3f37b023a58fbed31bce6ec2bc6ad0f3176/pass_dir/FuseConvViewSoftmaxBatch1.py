import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Match the conv2d + view(1, 1, -1) + softmax pattern for batch size 1"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.view(1, 1, -1)
    tmp_4 = tmp_3.softmax(dim=-1)
    return (tmp_4,)

def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for fused kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def batch1_conv_softmax_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    num_input_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for batch size 1"""
    
    # For batch size 1, each program handles one spatial location
    program_id = tl.program_id(0)
    spatial_size = height * width
    
    if program_id >= spatial_size:
        return
    
    h = program_id // width
    w = program_id % width
    
    # Apply 1x1 convolution: weighted sum of channels at spatial location
    conv_result = 0.0
    for c in range(num_input_channels):
        # Input offset for [1, num_input_channels, height, width] 
        input_offset = c * height * width + h * width + w
        input_val = tl.load(input_ptr + input_offset)
        
        # Weight offset for channel c in [1, num_input_channels, 1, 1]
        weight_offset = c
        weight_val = tl.load(weight_ptr + weight_offset)
        
        conv_result += input_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr)
    conv_result += bias_val
    
    # Store result directly - softmax will be done separately
    output_offset = h * width + w  # [1, 1, height, width] flattened
    tl.store(output_ptr + output_offset, conv_result)

@triton.jit
def batch1_softmax_kernel(
    input_ptr,
    output_ptr,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple softmax for batch size 1"""
    program_id = tl.program_id(0)
    
    if program_id >= spatial_size:
        return
    
    # For batch size 1, we can just compute softmax directly
    # Load all values for max computation
    max_val = -float('inf')
    for i in range(spatial_size):
        val = tl.load(input_ptr + i)
        if val > max_val:
            max_val = val
    
    # Compute sum of exp
    exp_sum = 0.0
    for i in range(spatial_size):
        val = tl.load(input_ptr + i)
        exp_val = tl.exp(val - max_val)
        exp_sum += exp_val
    
    # Store softmax result
    input_val = tl.load(input_ptr + program_id)
    softmax_val = tl.exp(input_val - max_val) / exp_sum
    tl.store(output_ptr + program_id, softmax_val)

@torch.fx.wrap  
def fused_batch1_conv_softmax(bias, weight, input_tensor):
    """Fused spatial attention for batch size 1"""
    
    # Get tensor shapes
    num_input_channels = input_tensor.shape[1] 
    height = input_tensor.shape[2]
    width = input_tensor.shape[3]
    spatial_size = height * width
    
    # Step 1: Apply conv2d
    conv_output = torch.zeros(1, spatial_size, dtype=input_tensor.dtype, device=input_tensor.device)
    
    block_size = 1024
    num_programs = (spatial_size + block_size - 1) // block_size
    
    batch1_conv_softmax_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight.view(-1),
        bias_ptr=bias,
        output_ptr=conv_output,
        num_input_channels=num_input_channels,
        height=height,
        width=width,
        BLOCK_SIZE=block_size
    )
    
    # Step 2: Apply softmax
    softmax_output = torch.zeros_like(conv_output)
    num_programs_softmax = (spatial_size + block_size - 1) // block_size
    
    batch1_softmax_kernel[(num_programs_softmax,)](
        input_ptr=conv_output,
        output_ptr=softmax_output,
        spatial_size=spatial_size,
        BLOCK_SIZE=block_size
    )
    
    return softmax_output.view(1, 1, spatial_size)

def replacement_func():
    """Return the fused kernel for batch size 1"""
    return fused_batch1_conv_softmax