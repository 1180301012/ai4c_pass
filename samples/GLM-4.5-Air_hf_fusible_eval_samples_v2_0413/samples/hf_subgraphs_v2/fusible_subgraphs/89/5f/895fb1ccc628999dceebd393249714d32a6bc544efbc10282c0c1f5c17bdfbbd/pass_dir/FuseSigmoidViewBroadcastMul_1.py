import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_sigmoid_view_broadcast_mul_kernel(
    sigmoid_input_ptr,
    input_tensor_ptr,
    output_ptr,
    sigmoid_batch_size,
    input_channels,
    input_height,
    input_width,
    CHANNELS: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles spatial coordinates (x, y)
    x = pid % input_width
    y = (pid // input_width) % input_height
    
    # Load sigmoid input - compile-time constant channels (512)
    channel_offset = tl.arange(0, CHANNELS)
    sigmoid_vals = tl.load(sigmoid_input_ptr + channel_offset)
    
    # Compute sigmoid using Triton operations - need fp32 precision for math
    sigmoid_vals_fp32 = sigmoid_vals.to(tl.float32)
    sigmoid_exp = tl.exp(-sigmoid_vals_fp32)
    sigmoid_computed_fp32 = 1.0 / (1.0 + sigmoid_exp)
    
    # Broadcast to spatial dimensions for all channels
    batch_idx = 0
    
    # For this spatial location (x,y), process all channels
    channel_idx = tl.arange(0, CHANNELS)
    
    # Load input tensor elements at this spatial location
    input_indices = (batch_idx * CHANNELS * input_height * input_width + 
                    channel_idx * input_height * input_width + 
                    y * input_width + x)
    input_vals = tl.load(input_tensor_ptr + input_indices)
    
    # Apply fused computation: input * sigmoid(broadcasted)
    # Convert sigmoid result back to input dtype for multiplication
    sigmoid_computed = sigmoid_computed_fp32.to(input_vals.type)
    output_vals = input_vals * sigmoid_computed
    
    # Store results
    output_indices = input_indices
    tl.store(output_ptr + output_indices, output_vals)

@torch.fx.wrap
def fused_sigmoid_view_broadcast_mul(in_0, in_1):
    # Get tensor shapes
    sigmoid_shape = in_0.shape  # [1, 512]
    input_shape = in_1.shape    # [1, 512, 64, 64]
    
    batch_size, channels = sigmoid_shape
    _, in_channels, height, width = input_shape
    
    # Note: assuming channels match (512 == 512)
    if channels != in_channels:
        raise ValueError(f"Channel dimension mismatch: {channels} vs {in_channels}")
    
    # Use a more efficient grid/block configuration
    # Each program handles one spatial location, processes all channels
    grid_size = height * width  # One program per (x,y) position
    BLOCK_SIZE = channels       # Process all channels in one block
    
    # Output tensor
    output = torch.empty_like(in_1)
    
    # Launch kernel with optimized configuration
    fused_sigmoid_view_broadcast_mul_kernel[(grid_size,)](
        sigmoid_input_ptr=in_0,
        input_tensor_ptr=in_1,
        output_ptr=output,
        sigmoid_batch_size=batch_size,
        input_channels=in_channels,
        input_height=height,
        input_width=width,
        CHANNELS=512,  # Fixed channel size from weight metadata
        BLOCK_SIZE_X=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_sigmoid_view_broadcast_mul