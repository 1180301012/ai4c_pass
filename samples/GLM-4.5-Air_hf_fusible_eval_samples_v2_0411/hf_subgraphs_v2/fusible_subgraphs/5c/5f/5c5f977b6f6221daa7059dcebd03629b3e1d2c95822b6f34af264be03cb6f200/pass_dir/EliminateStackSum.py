import torch
import triton
import triton.language as tl

# This pass matches the pattern where stack+sum operations are used redundantly
# pattern: torch.stack([tensor], dim=0).sum(dim=0) is equivalent to just tensor
# We need to match the full sequence including the conv2d and concatenation

def pattern(in_0, in_1, in_2, in_3):
    # Exact pattern matching from the model:
    # in_0 = bias, in_1 = weight, in_2 = conv_input, in_3 = concat_input
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.stack([tmp_2], dim=0)
    tmp_4 = tmp_3.sum(dim=0)
    tmp_5 = torch.cat([tmp_4, in_3], 1)
    return tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Triton kernel for optimized 1x1 convolution (pointwise convolution)
@triton.jit
def optimized_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_size, channels_out, channels_in, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size * channels_out:
        return
    
    # Calculate output position
    batch = pid // channels_out
    channel_out = pid % channels_out
    
    # Create masks for batch and channel bounds
    batch_mask = batch < batch_size
    channel_mask = (channel_out // channels_out) == 0
    
    if not batch_mask or not channel_mask:
        return
    
    # For 1x1 conv, process spatial positions sequentially per program
    total_spatial_positions = height * width
    
    # Check bounds for spatial positions
    h_start = 0
    h_end = height
    w_start = 0
    w_end = width
    
    # Process each spatial position
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            # Compute convolution result for this spatial position
            result = 0.0
            
            for c_in in range(channels_in):
                # Load input value with bounds checking
                input_idx = batch * channels_in * height * width + c_in * height * width + h * width + w
                input_val = tl.load(input_ptr + input_idx, mask=c_in < channels_in, other=0.0)
                
                # Load weight (always in bounds for this case)
                weight_idx = channel_out * channels_in + c_in
                weight_val = tl.load(weight_ptr + weight_idx, mask=True, other=0.0)
                
                result += input_val * weight_val
            
            # Add bias
            bias_val = tl.load(bias_ptr + channel_out, mask=True, other=0.0)
            result += bias_val
            
            # Store result
            output_idx = batch * channels_out * height * width + channel_out * height * width + h * width + w
            tl.store(output_ptr + output_idx, result, mask=(h < height) & (w < width))

# Triton kernel for tensor concatenation along channel dimension (dim=1)
@triton.jit
def optimized_concat_kernel(
    tensor1_ptr, tensor2_ptr,
    output_ptr,
    batch_size, channels1, channels2, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size * height * width:
        return
    
    # Calculate position
    batch = pid // (height * width)
    spatial = pid % (height * width)
    h = spatial // width
    w = spatial % width
    
    # Create spatial mask
    spatial_mask = (h < height) & (w < width) & (batch < batch_size)
    
    # Process all channels in this spatial position
    for c_out in range(channels1 + channels2):
        if c_out < channels1:
            # Copy from tensor1
            input_idx = batch * channels1 * height * width + c_out * height * width + h * width + w
            output_idx = batch * (channels1 + channels2) * height * width + c_out * height * width + h * width + w
            val = tl.load(tensor1_ptr + input_idx, mask=c_out < channels1, other=0.0)
            tl.store(output_ptr + output_idx, val, mask=spatial_mask & (c_out < channels1))
        else:
            # Copy from tensor2
            c2 = c_out - channels1
            input_idx = batch * channels2 * height * width + c2 * height * width + h * width + w
            output_idx = batch * (channels1 + channels2) * height * width + c_out * height * width + h * width + w
            val = tl.load(tensor2_ptr + input_idx, mask=c2 < channels2, other=0.0)
            tl.store(output_ptr + output_idx, val, mask=spatial_mask & (c_out >= channels1) & (c2 < channels2))

@torch.fx.wrap
def optimized_forward(in_0, in_1, in_2, in_3):
    # Get tensor shapes - bias, weight, conv_input, concat_input
    batch_size_conv, channels_in, height, width = in_2.shape
    channels_out = in_1.shape[0]
    batch_size_concat, channels_other, _, _ = in_3.shape
    
    # Ensure batch sizes match
    assert batch_size_conv == batch_size_concat, "Batch sizes must match"
    
    # Optimized conv2d computation (eliminates redundant stack-sum operations)
    conv_result = torch.empty((batch_size_conv, channels_out, height, width), 
                             dtype=in_2.dtype, device=in_2.device)
    
    # Launch optimized conv2d kernel
    total_conv_elements = batch_size_conv * channels_out
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(total_conv_elements, BLOCK_SIZE)
    
    optimized_conv2d_kernel[(grid_size,)](
        in_2, in_1, in_0, conv_result,
        batch_size_conv, channels_out, channels_in, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Optimized concatenation along channel dimension
    final_result = torch.empty((batch_size_concat, channels_out + channels_other, height, width), 
                              dtype=conv_result.dtype, device=conv_result.device)
    
    # Launch concat kernel
    total_spatial_elements = batch_size_concat * height * width
    concat_grid_size = triton.cdiv(total_spatial_elements, BLOCK_SIZE)
    
    optimized_concat_kernel[(concat_grid_size,)](
        conv_result, in_3, final_result,
        batch_size_concat, channels_out, channels_other, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return final_result

def replacement_func():
    return optimized_forward