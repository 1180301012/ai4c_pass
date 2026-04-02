import torch
import triton
import triton.language as tl

def pattern(x):
    """Match the pattern: hardtanh -> adaptive_avg_pool2d -> view -> flatten"""
    tmp_0 = torch.nn.functional.hardtanh(x, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = tmp_1.view(x.size(0), -1)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3

def replacement_args(x):
    """Extract input tensor for replacement"""
    return (x,)

@triton.jit
def fused_pool_view_kernel(
    x_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr
):
    """Optimized kernel that fuses pooling, view, and flatten operations"""
    # Each program handles one element of the output (batch_size x channels)
    pid = tl.program_id(0)
    
    if pid >= batch_size * channels:
        return
    
    # Calculate which batch and channel this program handles
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Calculate input offset for this batch and channel
    input_offset = batch_idx * channels * height * width + channel_idx * height * width
    
    # Load the entire channel patch for this batch (height x width elements)
    channel_data = tl.load(
        x_ptr + input_offset,
        mask=input_offset + tl.arange(0, height * width) < batch_size * channels * height * width,
        other=0.0
    )
    
    # Compute mean using accumulator to reduce precision loss
    sum_val = tl.sum(channel_data.to(tl.float32), 0)
    mean_val = sum_val / (height * width)
    
    # Store the result at the corresponding output position
    output_offset = batch_idx * channels + channel_idx
    tl.store(out_ptr + output_offset, mean_val.to(x_ptr.type.dtype))

@torch.fx.wrap
def fused_pool_view_flatten(x):
    """Wrapper function that launches the optimized kernel"""
    batch_size, channels, height, width = x.shape
    
    # Output shape should be [batch_size, channels]
    out = torch.empty((batch_size, channels), dtype=x.dtype, device=x.device)
    
    grid = (batch_size * channels,)
    
    fused_pool_view_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width
    )
    
    return out

def replacement_func():
    """Return the fused kernel function"""
    return fused_pool_view_flatten