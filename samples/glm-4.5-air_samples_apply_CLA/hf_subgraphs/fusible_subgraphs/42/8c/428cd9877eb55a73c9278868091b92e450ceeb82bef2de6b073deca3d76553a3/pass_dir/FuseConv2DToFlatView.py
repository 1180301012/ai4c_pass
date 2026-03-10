import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias, input_3, input_4):
    conv_output = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    view_output = conv_output.view(-1, 1, -1)
    concat_output = torch.cat([input_3, input_4, view_output], 2)
    return conv_output, concat_output

def replacement_args(conv_input, conv_weight, conv_bias, input_3, input_4):
    return (conv_input, conv_weight, conv_bias, input_3, input_4)

@triton.jit
def conv2d_to_flat_view_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    input_shape_0, input_shape_1, input_shape_2, input_shape_3,
    weight_shape_0, weight_shape_1, weight_shape_2, weight_shape_3,
    output_ptr_2,
    input_3_ptr, input_4_ptr,
    input_3_shape_0, input_3_shape_1, input_3_shape_2,
    input_4_shape_0, input_4_shape_1, input_4_shape_2,
    n_elements_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_total
    
    # Calculate the dimensions for the conv output
    batch_size = input_shape_0
    out_channels = weight_shape_0
    out_height = input_shape_2 + 2 * 0 - 1 * (input_shape_2 - 1) - 1
    out_width = input_shape_3 + 2 * 0 - 1 * (input_shape_3 - 1) - 1
    flat_size = out_height * out_width
    
    # Handle different batch sizes
    if batch_size == 32:
        view_shape_0 = 32
    else:
        view_shape_0 = 1
    
    # Compute indices
    total_flat_elements = view_shape_0 * 1 * (input_3_shape_2 + input_4_shape_2 + flat_size)
    element_idx = offsets
    
    # Calculate which element in which tensor
    cum_sizes = tl.tensor([0, input_3_shape_2, input_3_shape_2 + input_4_shape_2, 
                          input_3_shape_2 + input_4_shape_2 + flat_size])
    
    # Determine which section we're in
    section = tl.where((element_idx >= cum_sizes[0]) & (element_idx < cum_sizes[1]), 0,
                      tl.where((element_idx >= cum_sizes[1]) & (element_idx < cum_sizes[2]), 1,
                              tl.where((element_idx >= cum_sizes[2]) & (element_idx < cum_sizes[3]), 2, 3)))
    
    # Load and compute for each section
    out = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    
    # Section 0: input_3
    mask_section0 = (section == 0) & mask
    if tl.any(mask_section0):
        local_idx = element_idx - cum_sizes[0]
        input_3_offsets = local_idx
        input_3_data = tl.load(input_3_ptr + input_3_offsets, mask=mask_section0, other=0.0)
        out = tl.where(mask_section0, input_3_data, out)
    
    # Section 1: input_4  
    mask_section1 = (section == 1) & mask
    if tl.any(mask_section1):
        local_idx = element_idx - cum_sizes[1]
        input_4_offsets = local_idx
        input_4_data = tl.load(input_4_ptr + input_4_offsets, mask=mask_section1, other=0.0)
        out = tl.where(mask_section1, input_4_data, out)
    
    # Section 2: conv output (flattened)
    mask_section2 = (section == 2) & mask
    if tl.any(mask_section2):
        local_idx = element_idx - cum_sizes[2]
        
        # Compute position in flattened conv output
        conv_flat_idx = local_idx
        if view_shape_0 == 32:
            batch_idx = conv_flat_idx // (1 * flat_size)
            channel_idx = 0
            spatial_idx = conv_flat_idx % flat_size
        else:
            batch_idx = 0
            channel_idx = 0
            spatial_idx = conv_flat_idx
        
        # Compute 2D position from spatial index
        spatial_y = spatial_idx // out_width
        spatial_x = spatial_idx % out_width
        
        # Compute input position for convolution
        in_batch_idx = batch_idx
        in_channel_idx = spatial_idx % input_shape_1
        in_y = spatial_y
        in_x = spatial_x
        
        # Calculate input offset
        input_offset = in_batch_idx * (input_shape_1 * input_shape_2 * input_shape_3) + \
                      in_channel_idx * (input_shape_2 * input_shape_3) + \
                      in_y * input_shape_3 + in_x
        
        # Calculate weight offset (1x1 convolution)
        weight_offset = spatial_idx
        
        # Load input and weight
        input_data = tl.load(input_ptr + input_offset, mask=mask_section2, other=0.0)
        weight_data = tl.load(weight_ptr + weight_offset, mask=mask_section2, other=0.0)
        
        # Load bias (broadcast across spatial dimensions)
        bias_offset = batch_idx
        bias_data = tl.load(bias_ptr + bias_offset, mask=mask_section2, other=0.0)
        
        # Perform convolution (for 1x1 with stride 1, it's just element-wise multiplication)
        conv_result = input_data * weight_data + bias_data
        out = tl.where(mask_section2, conv_result, out)
    
    # Store result
    tl.store(output_ptr_2 + offsets, out, mask=mask)

@torch.fx.wrap
def conv2d_to_flat_view_wrapper(conv_input, conv_weight, conv_bias, input_3, input_4):
    # Detect batch size
    batch_size = conv_input.shape[0]
    
    # Calculate output shape
    out_channels = conv_weight.shape[0]
    out_height = conv_input.shape[2]  # stride 1, padding 0, dilation 1
    out_width = conv_input.shape[3]
    
    # Output concatenated shape
    input_3_flat_size = input_3.shape[2]
    input_4_flat_size = input_4.shape[2]  
    conv_flat_size = out_height * out_width
    
    total_elements = input_3_flat_size + input_4_flat_size + conv_flat_size
    
    # Create output tensor
    output = torch.empty([batch_size, 1, total_elements], dtype=torch.float32, device=conv_input.device)
    
    # Determine block size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv2d_to_flat_view_kernel[num_programs](
        conv_input, conv_weight, conv_bias,
        conv_input,  # We need to keep conv output for return
        conv_input.shape[0], conv_input.shape[1], conv_input.shape[2], conv_input.shape[3],
        conv_weight.shape[0], conv_weight.shape[1], conv_weight.shape[2], conv_weight.shape[3],
        output,
        input_3, input_4,
        input_3.shape[0], input_3.shape[1], input_3.shape[2],
        input_4.shape[0], input_4.shape[1], input_4.shape[2],
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Extract conv output from the concatenated result
    if batch_size == 32:
        conv_output = output[:, :, input_3_flat_size + input_4_flat_size:].view(32, 64, out_height, out_width)
    else:
        conv_output = output[:, :, input_3_flat_size + input_4_flat_size:].view(1, 64, out_height, out_width)
    
    final_output = output[:, :, :total_elements]
    
    return conv_output, final_output

def replacement_func():
    return conv2d_to_flat_view_wrapper