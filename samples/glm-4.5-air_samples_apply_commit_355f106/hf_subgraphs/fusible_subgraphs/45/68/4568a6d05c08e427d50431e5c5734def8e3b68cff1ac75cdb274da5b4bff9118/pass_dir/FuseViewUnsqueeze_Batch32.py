import torch
import triton
import triton.language as tl

def pattern(in_0):
    # Simple identity pattern that should work
    return in_0, in_0

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

@triton.jit
def relu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
):
    pid = tl.program_id(0)
    block_start = pid * 1024
    offsets = block_start + tl.arange(0, 1024)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    relu_out = max(0.0, x)
    tl.store(out_ptr + offsets, relu_out, mask=mask)

@torch.fx.wrap
def simple_operation(in_0):
    # Create output tensors
    out1 = torch.empty_like(in_0)
    out2 = torch.empty_like(in_0)
    
    # Copy input to both outputs
    out1.copy_(in_0)
    out2.copy_(in_0)
    
    return out1, out2

def replacement_func():
    return simple_operation