import torch
import triton
import triton.language as tl

@triton.jit
def final_transpose_reshape_kernel(
    input_ptr, output_ptr,
    input_batch, input_channels, input_height, input_width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Total elements in the input tensor
    total_input_elements = input_batch * input_channels * input_height * input_width
    
    if pid >= total_input_elements:
        return
    
    # Calculate indices in the input tensor (1, C, H, W)
    out_0 = pid // (input_channels * input_height * input_width)
    remainder = pid % (input_channels * input_height * input_width)
    
    channel = remainder // (input_height * input_width)
    remainder = remainder % (input_height * input_width)
    
    h = remainder // input_width
    w = remainder % input_width
    
    # Apply transpose(1, 2): (1, C, H, W) -> (1, H, C, W)
    # So we swap channel and height dimensions
    new_channel = h
    new_h = channel
    
    # Calculate new linear index for transposed tensor
    transposed_ptr_offset = out_0 * (input_channels * input_height * input_width) + \
                           new_channel * (input_height * input_width) + \
                           new_h * input_width + w
    
    # Load input value
    input_val = tl.load(input_ptr + pid, mask=(pid < total_input_elements), other=0.0)
    
    # Store in transposed position
    if transposed_ptr_offset < total_input_elements:
        tl.store(output_ptr + transposed_ptr_offset, input_val)

def pattern(tmp_9):
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, tmp_10.shape[1] * tmp_10.shape[2], tmp_10.shape[3])
    return tmp_11, tmp_10

def replacement_args(tmp_9):
    return (tmp_9,)

@torch.fx.wrap
def optimized_final_transpose_reshape(input_tensor):
    # Input shape after previous operations
    input_batch, input_channels, input_height, input_width = input_tensor.shape
    
    # Create output tensor for transposed result
    output = torch.empty((input_batch, input_channels, input_height, input_width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Triton launch configuration
    BLOCK_SIZE = 1024
    total_elements = input_batch * input_channels * input_height * input_width
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Execute transpose operation using Triton kernel
    final_transpose_reshape_kernel[grid](
        input_tensor,
        output,
        input_batch, input_channels, input_height, input_width,
        BLOCK_SIZE
    )
    
    # Now manually implement the reshape using the transposed tensor
    # The transposed tensor has shape (1, input_height, input_channels, input_width)
    # We want to reshape to (1, input_height * input_channels, input_width)
    
    transposed = output
    batch_size, transposed_h, transposed_c, transposed_w = transposed.shape
    
    # Create final reshaped output
    final_height = transposed_h * transposed_c
    final_width = transposed_w
    
    final_output = torch.empty((batch_size, final_height, final_width),
                              dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Copy data from transposed to reshaped (manually implement reshape)
    for i in range(batch_size):
        for h in range(transposed_h):
            for c in range(transposed_c):
                for w in range(transposed_w):
                    src_idx = i * (transposed_h * transposed_c * transposed_w) + \
                             h * (transposed_c * transposed_w) + \
                             c * transposed_w + w
                    dst_idx = i * (final_height * final_width) + \
                             (h * transposed_c + c) * final_width + w
                    
                    if src_idx < transposed.numel() and dst_idx < final_output.numel():
                        final_output.view(-1)[dst_idx] = transposed.view(-1)[src_idx]
    
    return final_output, transposed

def replacement_func():
    return optimized_final_transpose_reshape