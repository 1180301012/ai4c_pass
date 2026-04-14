import torch
import triton
import triton.language as tl

def pattern(in_4):
    # Reshape operation
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    # AvgPool2D operation
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    return tmp_5

def replacement_args(in_4):
    return (in_4,)

@triton.jit
def fused_reshape_avgpool_kernel(
    input_ptr,
    output_ptr,
    input_batch,
    input_channels,
    input_height,
    input_width,
    output_channels,
    output_height,
    output_width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global position
    pid = tl.program_id(0)
    num_programs = tl.cdiv(output_channels * output_height * output_width, BLOCK_SIZE)
    
    if pid >= num_programs:
        return
    
    # Calculate output coordinates for this program
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, output_channels * output_height * output_width)
    
    for idx in range(start_idx, end_idx):
        # Convert linear index to output coordinates
        c_out = idx // (output_height * output_width)
        h_out = (idx % (output_height * output_width)) // output_width
        w_out = idx % output_width
        
        # Calculate corresponding input coordinates
        # Since we're fusing reshape + avg_pool, the input tensor is reshaped first
        # Input shape: [input_batch, input_channels, input_height, input_width]
        # Reshaped shape: [1, 512, 16, 16]
        # AvgPool output: [1, 512, 8, 8]
        
        # In the fused operation, we directly compute from original input to avg_pool output
        # The reshape operation is virtualized since the data layout is handled implicitly
        
        # Calculate input region for averaging
        h_start = h_out * stride
        h_end = min(h_start + kernel_size, input_height)
        w_start = w_out * stride  
        w_end = min(w_start + kernel_size, input_width)
        
        # Sum elements in the kernel region
        sum_val = 0.0
        count = 0
        for h_in in range(h_start, h_end):
            for w_in in range(w_start, w_end):
                # Calculate input channel index after reshape
                # Original input: [4, 128, 256]
                # Reshaped: [1, 512, 16, 16] where 512 = 4 * 128
                input_channel_idx = c_out % input_channels
                batch_idx = c_out // input_channels  # Should be 0 due to reshape
                
                # Calculate linear index in original input
                linear_idx = batch_idx * input_channels * input_height * input_width + \
                           input_channel_idx * input_height * input_width + h_in * input_width + w_in
                
                input_val = tl.load(input_ptr + linear_idx, mask=(linear_idx < input_batch * input_channels * input_height * input_width), other=0.0).to(tl.float32)
                sum_val += input_val
                count += 1
        
        # Compute average
        if count > 0:
            avg_val = sum_val / count
        else:
            avg_val = 0.0
        
        # Store result
        output_idx = c_out * output_height * output_width + h_out * output_width + w_out
        tl.store(output_ptr + output_idx, avg_val, mask=(idx < output_channels * output_height * output_width))

@torch.fx.wrap  
def fused_reshape_avgpool(input_tensor):
    # Get input tensor shape [4, 128, 256]
    input_batch, input_channels, input_height, input_width = 4, 128, 256, 256
    
    # Target reshape shape [1, 512, 16, 16]  
    reshaped_batch, reshaped_channels, reshaped_height, reshaped_width = 1, 512, 16, 16
    
    # AvgPool2D parameters
    kernel_size = stride = 2
    padding = 0
    
    # Output shape after avg_pool2d: [1, 512, 8, 8]
    output_batch, output_channels, output_height, output_width = 1, 512, 8, 8
    
    # Calculate total output elements
    total_elements = output_channels * output_height * output_width
    
    # Use optimal block size
    BLOCK_SIZE = 1024
    
    # Number of programs needed
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty((output_batch, output_channels, output_height, output_width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    fused_reshape_avgpool_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        input_batch=input_batch,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        output_channels=output_channels,
        output_height=output_height,
        output_width=output_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_reshape_avgpool