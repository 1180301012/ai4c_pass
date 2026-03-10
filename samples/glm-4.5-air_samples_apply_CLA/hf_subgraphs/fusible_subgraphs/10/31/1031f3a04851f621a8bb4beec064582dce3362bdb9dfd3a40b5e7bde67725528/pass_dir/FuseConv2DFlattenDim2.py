import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias):
    tmp_2 = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(tmp_2, 2)
    return tmp_3

def replacement_args(conv_input, conv_weight, conv_bias):
    return (conv_input, conv_weight, conv_bias)

@triton.jit
def optimized_conv2d_flatten_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_batch,
    n_input_channels,
    input_height,
    input_width,
    weight_output_channels,
):
    # Each program handles one spatial location (i,j) for all batches and output channels
    spatial_program_id = tl.program_id(0)
    
    # Decode spatial coordinates
    spatial_i = spatial_program_id // input_width
    spatial_j = spatial_program_id % input_width
    
    # Check if spatial coordinates are within bounds
    mask = (spatial_i < input_height) & (spatial_j < input_width)
    
    if not mask:
        return
    
    # Flatten spatial coordinate
    spatial_offset = spatial_i * input_width + spatial_j
    
    # Process all batches and output channels for this spatial location
    for b in range(n_batch):
        for oc in range(weight_output_channels):
            # Input base offset for this batch
            input_batch_base = b * n_input_channels * input_height * input_width
            
            # Weight offset for this output channel (1x1 convolution)
            weight_channel_base = oc * n_input_channels
            
            # Output offset for this batch and output channel
            output_offset = b * weight_output_channels * input_height * input_width + oc * input_height * input_width + spatial_offset
            
            # Load bias for this output channel
            bias_val = tl.load(bias_ptr + oc)
            
            # Compute 1x1 convolution: sum(input[b,:,i,j] * weight[oc,:,0,0])
            conv_result = 0.0
            for ic in range(n_input_channels):
                # Input tensor offset: [b][ic][i][j]
                input_tensor_offset = input_batch_base + ic * input_height * input_width + spatial_offset
                
                # Weight tensor offset: [oc][ic] (assuming flattened 17x160 weights)
                weight_tensor_offset = weight_channel_base + ic
                
                # Load values
                input_val = tl.load(input_ptr + input_tensor_offset)
                weight_val = tl.load(weight_ptr + weight_tensor_offset)
                
                conv_result += input_val * weight_val
            
            # Store result: conv_result + bias
            final_result = conv_result + bias_val
            tl.store(output_ptr + output_offset, final_result)

@torch.fx.wrap  
def optimized_conv2d_flatten(input, weight, bias):
    # Get tensor properties
    n_batch, n_input_channels, input_height, input_width = input.shape
    weight_output_channels, n_weight_channels, weight_height, weight_width = weight.shape
    
    # Calculate output tensor shape
    output_total_dims = input_height * input_width
    output_shape = (n_batch, weight_output_channels, output_total_dims)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Set up grid dimensions - one program per spatial location
    spatial_grid_size = input_height * input_width
    
    # Launch kernel
    optimized_conv2d_flatten_kernel[(spatial_grid_size,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_batch=n_batch,
        n_input_channels=n_input_channels,
        input_height=input_height,
        input_width=input_width,
        weight_output_channels=weight_output_channels,
    )
    
    return output

def replacement_func():
    return optimized_conv2d_flatten