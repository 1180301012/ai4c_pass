import torch
import triton
import triton.language as tl

@triton.jit
def fused_avg_pool2d_flatten_kernel(
    x_ptr,                      # input: [1, C, H, W]
    out_ptr,                    # output: [1, C]
    C: tl.constexpr,            # channels (2048)
    H: tl.constexpr,            # spatial height
    W: tl.constexpr,            # spatial width
):
    # Each program handles one channel
    c = tl.program_id(0)
    
    # Check if channel is valid
    if c >= C:
        return
    
    # Calculate starting position for this channel in input tensor
    base_offset = c * (H * W)
    
    # Load all elements for this channel and compute sum
    # Use next power of 2 (256) for arange size to satisfy Triton requirement
    channel_offset = base_offset + tl.arange(0, 256)
    mask = channel_offset < (base_offset + H * W)
    
    # Load all elements for this channel
    channel_elements = tl.load(x_ptr + channel_offset, mask=mask, other=0.0)
    
    # Compute average (sum / count)
    channel_sum = tl.sum(channel_elements)
    channel_avg = channel_sum * (1.0 / float(H * W))  # Use multiplication instead of division
    
    # Store result at corresponding position in output
    out_offset = c
    out_mask = out_offset < C
    tl.store(out_ptr + out_offset, channel_avg, mask=out_mask)



@torch.fx.wrap  
def fused_adaptive_avg_pool2d_flatten(x):
    """
    Fused adaptive average pooling (to 1x1) followed by flatten.
    Input: [1, C, H, W] -> Output: [1, C]
    """
    batch, channels, height, width = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    
    # Output is [1, channels]
    out = torch.empty(batch, channels, dtype=x.dtype, device=x.device)
    
    # Launch kernel - one program per channel
    fused_avg_pool2d_flatten_kernel[(channels,)](
        x_ptr=x,
        out_ptr=out,
        C=channels,
        H=height,
        W=width,
    )
    
    return out

def pattern(input_tensor):
    """Match adaptive_avg_pool2d(..., 1) followed by flatten(1, -1)"""
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(input_tensor, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7

def replacement_args(input_tensor):
    return (input_tensor,)

def replacement_func():
    return fused_adaptive_avg_pool2d_flatten