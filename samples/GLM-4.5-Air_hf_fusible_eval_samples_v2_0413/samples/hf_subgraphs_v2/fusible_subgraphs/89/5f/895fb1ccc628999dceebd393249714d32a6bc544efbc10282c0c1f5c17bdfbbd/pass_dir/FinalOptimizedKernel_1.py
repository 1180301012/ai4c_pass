import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    tmp_4 = torch.relu_(tmp_3)
    return tmp_4

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_fused_kernel(
    sigmoid_input_ptr,
    input_tensor_ptr, 
    output_ptr,
    input_channels,
    input_height,
    input_width,
    CHANNELS: tl.constexpr,
):
    pid = tl.program_id(0)
    x = pid % input_width
    y = (pid // input_width) % input_height
    
    # Load and compute sigmoid efficiently
    channel_idx = tl.arange(0, CHANNELS)
    sigmoid_vals = tl.load(sigmoid_input_ptr + channel_idx)
    sigmoid_exp = tl.exp(-sigmoid_vals.to(tl.float32))
    sigmoid_computed = 1.0 / (1.0 + sigmoid_exp)
    
    # Load input data and compute result in one pass
    batch_idx = 0
    input_indices = (batch_idx * CHANNELS * input_height * input_width + 
                    channel_idx * input_height * input_width + 
                    y * input_width + x)
    input_vals = tl.load(input_tensor_ptr + input_indices)
    
    # Fused computation: input * (1 + sigmoid)
    result = input_vals * (1.0 + sigmoid_computed)
    
    # Apply ReLU
    output_vals = tl.maximum(result, 0.0)
    
    tl.store(output_ptr + input_indices, output_vals)

@torch.fx.wrap
def final_optimized_computation(in_0, in_1):
    # Extract tensor dimensions
    _, channels = in_0.shape
    _, _, height, width = in_1.shape
    
    # Grid configuration: one program per spatial location
    grid_size = height * width
    
    # Output tensor
    output = torch.empty_like(in_1)
    
    # Launch optimized kernel
    optimized_fused_kernel[(grid_size,)](
        sigmoid_input_ptr=in_0,
        input_tensor_ptr=in_1,
        output_ptr=output,
        input_channels=channels,
        input_height=height,
        input_width=width,
        CHANNELS=512,
    )
    
    return output

def replacement_func():
    return final_optimized_computation