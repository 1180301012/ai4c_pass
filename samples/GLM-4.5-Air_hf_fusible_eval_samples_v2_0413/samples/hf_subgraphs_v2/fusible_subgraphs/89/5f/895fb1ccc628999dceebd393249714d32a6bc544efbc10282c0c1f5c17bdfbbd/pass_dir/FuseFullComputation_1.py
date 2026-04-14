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
def full_fused_kernel(
    sigmoid_input_ptr,
    input_tensor_ptr,
    output_ptr,
    sigmoid_batch_size,
    input_channels,
    input_height,
    input_width,
    CHANNELS: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles spatial coordinates (x, y)
    x = pid % input_width
    y = (pid // input_width) % input_height
    
    # Load sigmoid input and compute sigmoid once for all channels
    channel_offset = tl.arange(0, CHANNELS)
    sigmoid_vals = tl.load(sigmoid_input_ptr + channel_offset)
    
    # Compute sigmoid in fp32 for precision
    sigmoid_vals_fp32 = sigmoid_vals.to(tl.float32)
    sigmoid_exp = tl.exp(-sigmoid_vals_fp32)
    sigmoid_computed_fp32 = 1.0 / (1.0 + sigmoid_exp)
    
    # Precompute scaling factor (1 + sigmoid_val) for fused operation  
    scale_factors_fp32 = 1.0 + sigmoid_computed_fp32
    scale_factors = scale_factors_fp32
    
    # Load input tensor elements at this spatial location for all channels
    batch_idx = 0
    input_indices = (batch_idx * CHANNELS * input_height * input_width + 
                    channel_offset * input_height * input_width + 
                    y * input_width + x)
    input_vals = tl.load(input_tensor_ptr + input_indices)
    
    # Apply fused computation: input * (1 + sigmoid) - this does tmp_2 + tmp_3 together
    scaled_vals = input_vals * scale_factors
    
    # Apply ReLU operation
    relu_vals = tl.maximum(scaled_vals, 0.0)
    
    # Store results
    output_indices = input_indices
    tl.store(output_ptr + output_indices, relu_vals)

@torch.fx.wrap
def full_fused_computation(in_0, in_1):
    # Get tensor shapes
    sigmoid_shape = in_0.shape  # [1, 512]
    input_shape = in_1.shape    # [1, 512, 64, 64]
    
    batch_size, channels = sigmoid_shape
    _, in_channels, height, width = input_shape
    
    # Note: assuming channels match (512 == 512)
    if channels != in_channels:
        raise ValueError(f"Channel dimension mismatch: {channels} vs {in_channels}")
    
    # Use one program per spatial location, process all channels
    grid_size = height * width
    BLOCK_SIZE = channels
    
    # Output tensor
    output = torch.empty_like(in_1)
    
    # Launch kernel
    full_fused_kernel[(grid_size,)](
        sigmoid_input_ptr=in_0,
        input_tensor_ptr=in_1,
        output_ptr=output,
        sigmoid_batch_size=batch_size,
        input_channels=in_channels,
        input_height=height,
        input_width=width,
        CHANNELS=512,  # Fixed channel size from weight metadata
        BLOCK_SIZE_Y=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return full_fused_computation