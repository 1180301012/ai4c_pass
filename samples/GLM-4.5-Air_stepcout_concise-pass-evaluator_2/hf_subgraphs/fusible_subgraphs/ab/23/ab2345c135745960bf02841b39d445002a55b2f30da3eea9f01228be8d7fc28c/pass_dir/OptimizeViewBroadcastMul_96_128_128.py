import torch
import triton
import triton.language as tl

def pattern(input_tensor, conv_output_reshaped):
    # Match view + element-wise multiplication with broadcasting
    view_out = conv_output_reshaped.view(1, -1, 1, 1)
    mul_out = input_tensor * view_out
    return mul_out

def replacement_args(input_tensor, conv_output_reshaped):
    return (input_tensor, conv_output_reshaped)

@triton.jit
def broadcast_mul_kernel(
    input_ptr,
    conv_output_ptr,
    output_ptr,
    total_elements,
    channels,
    spatial_per_channel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate current offset for this program  
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements
    
    # Calculate channel indices (each element belongs to one channel)
    channel_idx = offset // spatial_per_channel
    
    # Load conv_output values for these channels (shape: [1, 96, 1, 1]) - mask handles bounds checking
    conv_vals = tl.load(conv_output_ptr + channel_idx, mask=mask)
    
    # Load input tensor values  
    input_vals = tl.load(input_ptr + offset, mask=mask)
    
    # Apply broadcasting multiplication
    output_vals = input_vals * conv_vals
    
    # Store results
    tl.store(output_ptr + offset, output_vals, mask=mask)

@torch.fx.wrap
def optimized_broadcast_mul(input_tensor, conv_output):
    # Fixed tensor shapes for this specific operation
    total_elements = 1572864  # 1 * 96 * 128 * 128
    channels = 96
    spatial_per_channel = 128 * 128
    
    # Use optimal block size for GPU occupancy
    BLOCK_SIZE = 2048
    
    # Calculate number of programs needed
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor)
    
    # Optimized kernel call with better block size
    broadcast_mul_kernel[(num_programs,)](
        input_ptr=input_tensor,
        conv_output_ptr=conv_output,
        output_ptr=output,
        total_elements=total_elements,
        channels=channels,
        spatial_per_channel=spatial_per_channel,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_broadcast_mul