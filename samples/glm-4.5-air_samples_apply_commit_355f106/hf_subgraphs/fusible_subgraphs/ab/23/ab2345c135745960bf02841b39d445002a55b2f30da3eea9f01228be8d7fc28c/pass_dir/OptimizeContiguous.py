import torch
import triton
import triton.language as tl

@triton.jit
def optimized_multiplication_kernel(
    sigmoid_out_ptr,  # Sigmoid output [1, 96, 1, 1]
    input_ptr,        # Input tensor [1, 96, 128, 128]
    out_ptr,          # Output tensor [1, 96, 128, 128] (already contiguous)
    n_elements,       # Number of elements in output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load sigmoid output (broadcasted)
    # Since sigmoid_out is [1, 96, 1, 1], we need to broadcast it to output shape
    # Calculate position in output tensor
    total_elements = 96 * 128 * 128  # 1,572,864 elements
    channel_idx = offsets % total_elements // (128 * 128)  # Channel index (0-95)
    
    # Load sigmoid values (one per channel) with proper bounds checking
    sigmoid_values = tl.load(sigmoid_out_ptr + channel_idx, mask=channel_idx < 96)
    
    # Broadcast the sigmoid values to current thread block
    sigmoid_broadcasted = tl.broadcast_to(sigmoid_values, (BLOCK_SIZE,), )
    
    # Load input tensor values
    input_values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Perform multiplication
    out_values = sigmoid_broadcasted * input_values
    
    # Store result directly - output is already contiguous
    tl.store(out_ptr + offsets, out_values, mask=mask)

@torch.fx.wrap
def optimized_multiplication(sigmoid_out, input_tensor):
    # Determine output shape
    batch, channels, height, width = input_tensor.shape
    output_shape = (batch, channels, height, width)
    output_size = batch * channels * height * width
    
    # Create output tensor that is already contiguous
    out = torch.empty(output_shape, dtype=torch.float32, device=sigmoid_out.device)
    
    # Choose appropriate block size for optimal GPU utilization
    BLOCK_SIZE = 1024  # Good balance for most GPUs
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_multiplication_kernel[(num_programs,)](
        sigmoid_out_ptr=sigmoid_out,
        input_ptr=input_tensor,
        out_ptr=out,
        n_elements=output_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out  # Already contiguous, no need for .contiguous()

def pattern(sigmoid_out, input_tensor):
    # Original pattern: multiplication followed by contiguous
    # tmp_5 = input_tensor * sigmoid_out.view(1, -1, 1, 1)
    # tmp_6 = tmp_5.contiguous()  # This might be unnecessary
    tmp_5 = input_tensor * sigmoid_out.view(1, -1, 1, 1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(sigmoid_out, input_tensor):
    return (sigmoid_out, input_tensor)

def replacement_func():
    return optimized_multiplication