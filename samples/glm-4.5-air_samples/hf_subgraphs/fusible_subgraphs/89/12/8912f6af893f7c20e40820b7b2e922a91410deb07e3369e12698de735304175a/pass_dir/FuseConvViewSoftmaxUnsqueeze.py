import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # tmp_0 = in_0
    # tmp_1 = in_1
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    
    # tmp_3 = tmp_2.view(N, 1, L)
    tmp_3 = tmp_2.view(tmp_2.shape[0], 1, -1)
    tmp_2 = None
    
    # tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_3 = None
    
    # tmp_5 = tmp_4.unsqueeze(-1)
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = None
    
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def softmax_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute softmax (element-wise for each position in sequence)
    max_val = tl.max(x, axis=0, keepdims=True)
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x, axis=0, keepdims=True)
    y = exp_x / sum_exp
    
    # Store result
    tl.store(y_ptr + offsets, y, mask=mask)

@triton.jit
def conv2d_1x1_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial position
    pid = tl.program_id(0)
    h = pid // width
    w = pid % width
    
    # Calculate total elements per batch and channel
    elements_per_channel = height * width
    
    # Process all channels for this spatial position
    channel_offsets = tl.arange(0, channels)
    input_base = h * width + w
    
    # Load weights and bias
    weights = tl.load(weight_ptr + tl.arange(0, channels))
    bias = tl.load(bias_ptr)
    
    # Load input data for this spatial position across all channels
    input_offsets = input_base + channel_offsets * elements_per_channel
    input_data = tl.load(input_ptr + input_offsets, mask=input_offsets < batch_size * elements_per_channel, other=0.0)
    
    # Apply convolution (1x1, so just multiply and add)
    output_data = input_data * weights + bias
    
    # Store output
    output_offsets = input_base  # Same spatial layout for output
    tl.store(output_ptr + output_offsets, output_data, mask=output_offsets < batch_size * elements_per_channel)

@torch.fx.wrap
def optimized_conv_softmax_unsqueeze(conv_bias, conv_weight, conv_input):
    batch_size, in_channels, in_height, in_width = conv_input.shape
    
    # Step 1: Perform 1x1 convolution using Triton
    total_elements = batch_size * in_channels * in_height * in_width
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    conv_out = torch.empty_like(conv_input)
    conv2d_1x1_kernel[(num_programs,)](
        input_ptr=conv_input,
        weight_ptr=conv_weight,
        bias_ptr=conv_bias,
        output_ptr=conv_out,
        batch_size=batch_size,
        channels=in_channels,
        height=in_height,
        width=in_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Step 2: View operation - reshape for softmax
    view_shape = (batch_size, 1, -1)
    view_out = conv_out.view(view_shape)
    
    # Step 3: Apply optimized softmax using Triton
    batch_size, _, seq_len = view_out.shape
    total_seq_elements = batch_size * seq_len
    
    BLOCK_SIZE = 1024
    num_programs = (total_seq_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    softmax_out = torch.empty_like(view_out)
    softmax_kernel[(num_programs,)](
        x_ptr=view_out,
        y_ptr=softmax_out,
        n_elements=total_seq_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Step 4: Unsqueeze operation
    final_out = softmax_out.unsqueeze(-1)
    
    return final_out

def replacement_func():
    return optimized_conv_softmax_unsqueeze