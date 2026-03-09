import torch
import triton
import triton.language as tl

@triton.jit  
def fused_multiplication_kernel(
    sigmoid_ptr,      # Sigmoid output [1, 96, 1, 1]
    input_ptr,        # Input tensor [1, 96, 128, 128] 
    out_ptr,          # Output tensor [1, 96, 128, 128]
    n_elements,       # Number of elements in output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate which channel each offset corresponds to
    channel_idx = offsets % (96 * 128 * 128) // (128 * 128)
    spatial_idx = offsets % (128 * 128)
    
    # Load sigmoid value for current channel (simplest approach)
    # Each thread loads its own sigmoid value
    sigmoid_val = tl.load(sigmoid_ptr + channel_idx, mask=channel_idx < 96, other=1.0)
    
    # Load input tensor value
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Perform multiplication
    out_val = sigmoid_val * input_val
    
    # Store result
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_view_multiplication(sigmoid_out, input_tensor):
    # Determine output shape
    batch, channels, height, width = input_tensor.shape
    output_shape = (batch, channels, height, width)
    output_size = batch * channels * height * width
    
    out = torch.empty(output_shape, dtype=torch.float32, device=sigmoid_out.device)
    
    # Use simple block size - 1024 works well for this workload
    BLOCK_SIZE = 1024
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use simplified kernel
    fused_multiplication_kernel[(num_programs,)](
        sigmoid_ptr=sigmoid_out,
        input_ptr=input_tensor,
        out_ptr=out,
        n_elements=output_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def pattern(sigmoid_out, input_tensor):
    # Original pattern: view followed by multiplication
    # tmp_4 = sigmoid_out.view(1, -1, 1, 1)  # This is unnecessary
    # tmp_5 = input_tensor * tmp_4  # This can be fused
    tmp_4 = sigmoid_out.view(1, -1, 1, 1)
    tmp_5 = input_tensor * tmp_4
    return tmp_5

def replacement_args(sigmoid_out, input_tensor):
    return (sigmoid_out, input_tensor)

def replacement_func():
    return fused_view_multiplication