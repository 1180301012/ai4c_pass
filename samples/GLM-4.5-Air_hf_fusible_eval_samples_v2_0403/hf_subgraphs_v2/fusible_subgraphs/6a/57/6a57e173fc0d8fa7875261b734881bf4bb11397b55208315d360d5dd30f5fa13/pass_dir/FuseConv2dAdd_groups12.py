import torch
import triton
import triton.language as tl

def pattern(value_layer, weight_tensor, context_layer):
    # Conv2D with groups=12 (seen in some models)
    conv2d = torch.conv2d(value_layer, weight_tensor, None, (1, 1), (32, 0), (1, 1), 12)
    # Inplace addition to context layer
    context_layer += conv2d
    # Return the updated context layer as the observable result
    return context_layer

def replacement_args(value_layer, weight_tensor, context_layer):
    return (value_layer, weight_tensor, context_layer)

@triton.jit
def fused_conv2d_add_kernel_groups12(
    value_layer_ptr,
    weight_ptr,
    context_layer_ptr,
    batch_size,
    groups,
    seq_len,
    channel_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch*groups*seq_len dimension
    program_id = tl.program_id(0)
    total_programs = batch_size * groups * seq_len
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_programs
    
    # For conv2d with padding (32, 0), we need to handle the spatial dimensions carefully
    # Input tensor: [batch_size, groups, seq_len, 8]
    # Weight tensor: [12, 1, 65, 1]
    
    # Process each group separately for better parallelism
    group_id = program_id % groups
    batch_id = (program_id // groups) % batch_size
    seq_id = (program_id // (groups * batch_size)) % seq_len
    
    # Compute base addresses
    value_base = batch_id * groups * seq_len * 8 + group_id * seq_len * 8 + seq_id * 8
    context_base = batch_id * groups * seq_len * 8 + group_id * seq_len * 8 + seq_id * 8
    
    # Load input and context values
    value_ptr = value_layer_ptr + value_base
    context_ptr = context_layer_ptr + context_base
    
    values = tl.load(value_ptr + tl.arange(0, 8), mask=tl.arange(0, 8) < 8)
    context_values = tl.load(context_ptr + tl.arange(0, 8), mask=tl.arange(0, 8) < 8)
    
    # Convolution operation with padding (32, 0)
    conv_result = 0.0
    
    # For each position in the 65 kernel width, compute the convolution
    # With padding (32, 0), we start offset by 32 in the sequence dimension
    kernel_start = max(0, seq_id - 32)
    kernel_end = min(seq_len, seq_id + 65 - 32)
    
    for k in range(kernel_start, kernel_end):
        seq_offset = k - (seq_id - 32)
        if 0 <= seq_offset < 65:
            # Load weight value (weight has shape [12, 1, 65, 1])
            weight_offset = group_id * 65 + seq_offset
            weight_val = tl.load(weight_ptr + weight_offset, mask=(weight_offset < 12 * 65))
            
            # Load input value
            input_offset = batch_id * groups * seq_len * 8 + group_id * seq_len * 8 + k * 8
            input_ptr = value_layer_ptr + input_offset
            input_vals = tl.load(input_ptr + tl.arange(0, 8), mask=tl.arange(0, 8) < 8)
            
            # Accumulate convolution result
            conv_result += weight_val * input_vals[0]  # Using first channel (weight has only 1 channel)
    
    # Add conv result to context
    result = context_values + conv_result
    
    # Store back to context layer
    tl.store(context_ptr + tl.arange(0, 8), result, mask=tl.arange(0, 8) < 8)

@torch.fx.wrap  
def fused_conv2d_add_groups12(value_layer, weight_tensor, context_layer):
    batch_size, groups, seq_len, channel_size = value_layer.shape
    tensor_elements = batch_size * groups * seq_len * channel_size
    BLOCK_SIZE = 32  # Number of elements processed by each program
    
    num_programs = (tensor_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv2d_add_kernel_groups12[(num_programs,)](
        value_layer_ptr=value_layer,
        weight_ptr=weight_tensor,
        context_layer_ptr=context_layer,
        batch_size=batch_size,
        groups=groups,
        seq_len=seq_len,
        channel_size=channel_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return context_layer

def replacement_func():
    return fused_conv2d_add_groups12