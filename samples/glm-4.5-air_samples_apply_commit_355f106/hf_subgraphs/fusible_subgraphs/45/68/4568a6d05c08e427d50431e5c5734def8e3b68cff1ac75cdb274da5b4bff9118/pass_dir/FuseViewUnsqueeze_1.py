import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    batch_size, channels, height, width = in_0.shape
    tmp_1 = tmp_0.view(batch_size, channels, height * width)
    tmp_2 = tmp_1.unsqueeze(1)
    return tmp_2, tmp_0

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_kernel(
    in_0_ptr,
    out_ptr,
    tmp_0_ptr,  # ReLU output  
    n_elements,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
):
    # Each program handles one element
    pid = tl.program_id(0)
    
    # Calculate intermediate dimensions
    hw_size = height * width
    elements_per_batch = channels * height * width
    
    # Get batch, channel, and spatial indices
    batch_idx = pid // elements_per_batch
    remaining = pid % elements_per_batch
    channel_idx = remaining // (height * width)
    spatial_idx = remaining % (height * width)
    
    # Calculate output dimensions after unsqueeze
    output_elements_per_batch = channels * height * width  # No change from input for tmp_2
    
    # Input offset: [batch_idx, channel_idx, h, w]
    input_offset = pid
    
    # tmp_0 (ReLU output) offset - same as input for layout
    relu_offset = pid
    
    # tmp_2 (fused output) offset: [batch_idx, 0, channel_idx, h*w] 
    # The output has shape [batch_size, 1, channels, height*width]
    batch_offset_out = batch_idx * channels * hw_size  # channel dim = 1
    channel_offset_out = hw_size  # Extra dimension from unsqueeze
    spatial_offset_out = spatial_idx
    relu_offset_out = batch_offset_out + channel_offset_out + channel_idx * hw_size + spatial_offset_out
    
    mask = pid < n_elements
    
    if mask:
        # Load input and apply ReLU
        val = tl.load(in_0_ptr + input_offset, mask=mask)
        relu_val = max(0.0, val)
        
        # Store both results
        tl.store(tmp_0_ptr + relu_offset, relu_val, mask=mask)
        tl.store(out_ptr + relu_offset_out, relu_val, mask=mask)

@torch.fx.wrap
def fused_operation(in_0):
    # Get input shape
    batch_size, channels, height, width = in_0.shape
    hw_size = height * width
    
    # Create output tensors
    # tmp_0: same shape as input - this is the ReLU output
    tmp_0 = torch.empty_like(in_0)
    # tmp_2: [batch_size, 1, channels, hw_size] - this is the fused output
    output_shape = (batch_size, 1, channels, hw_size)
    tmp_2 = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    n_elements = in_0.numel()
    
    # Launch kernel
    # Use 1D grid for all elements
    grid = (n_elements + 1023) // 1024
    
    fused_kernel[grid](
        in_0_ptr=in_0,
        out_ptr=tmp_2,
        tmp_0_ptr=tmp_0, 
        n_elements=n_elements,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width
    )
    
    return tmp_2, tmp_0

def replacement_func():
    return fused_operation