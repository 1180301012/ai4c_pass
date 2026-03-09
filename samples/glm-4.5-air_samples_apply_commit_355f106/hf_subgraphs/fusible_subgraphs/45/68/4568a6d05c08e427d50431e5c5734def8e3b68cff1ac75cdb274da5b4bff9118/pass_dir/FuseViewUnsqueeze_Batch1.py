import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.view(1, 512, 4096)
    tmp_2 = tmp_1.unsqueeze(1)
    return tmp_2, tmp_0

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_kernel(
    in_0_ptr,
    out_ptr,
    tmp_0_ptr,
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
    
    # Input offset: [batch_idx, channel_idx, h, w]
    input_offset = pid
    
    # tmp_0 (ReLU output) offset - same as input for layout
    relu_offset = pid
    
    # tmp_2 (fused output) offset: [batch_idx, 0, channel_idx, h*w]
    output_elements_per_batch = 512 * 4096  # channels * height*width after view
    batch_offset_out = batch_idx * output_elements_per_batch
    channel_offset_out = 4096  # Extra dimension from unsqueeze and size after view
    spatial_offset_out = spatial_idx
    relu_out_offset = batch_offset_out + channel_offset_out + channel_idx * hw_size + spatial_offset_out
    
    mask = pid < n_elements
    
    if mask:
        # Load input and apply ReLU
        val = tl.load(in_0_ptr + input_offset, mask=mask)
        relu_val = max(0.0, val)
        
        # Store both results
        tl.store(tmp_0_ptr + relu_offset, relu_val, mask=mask)
        tl.store(out_ptr + relu_out_offset, relu_val, mask=mask)

@torch.fx.wrap
def fused_operation(in_0):
    # Get input shape
    batch_size, channels, height, width = in_0.shape
    
    # Create output tensors
    # tmp_0: same shape as input - this is the ReLU output  
    tmp_0 = torch.empty_like(in_0)
    # tmp_2: [batch_size, 1, 512, 4096] - this is the fused output
    output_shape = (batch_size, 1, 512, 4096)
    tmp_2 = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    n_elements = in_0.numel()
    
    # Launch kernel
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