import torch
import triton
import triton.language as tl

@triton.jit
def spatial_reduction_kernel(
    input_ptr,
    output_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    # Each program handles one channel
    channel_pid = tl.program_id(0)
    
    # Load spatial data for this channel: [H, W]
    channel_offset = channel_pid * H * W
    offsets = channel_offset + tl.arange(0, H * W)
    mask = offsets < (C * H * W)
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU
    relu_output = tl.maximum(input_data, 0.0)
    
    # Compute mean by summing and dividing
    channel_sum = tl.sum(relu_output)
    channel_mean = channel_sum / (H * W)
    
    # Store result at channel position
    tl.store(output_ptr + channel_pid, channel_mean)

@torch.fx.wrap
def spatial_reduction_optimization(input_tensor):
    N, C, H, W = input_tensor.shape
    output = torch.empty((N, C), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Process each sample separately
    for i in range(N):
        grid = (C,)
        spatial_reduction_kernel[grid](
            input_ptr=input_tensor[i],
            output_ptr=output[i],
            C=C,
            H=H,
            W=W,
        )
    
    return output

def pattern(x):
    """Match ReLU + adaptive avg pool + flatten"""
    tmp_5 = torch.nn.functional.relu(x)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7

def replacement_args(x):
    return (x,)

def replacement_func():
    return spatial_reduction_optimization