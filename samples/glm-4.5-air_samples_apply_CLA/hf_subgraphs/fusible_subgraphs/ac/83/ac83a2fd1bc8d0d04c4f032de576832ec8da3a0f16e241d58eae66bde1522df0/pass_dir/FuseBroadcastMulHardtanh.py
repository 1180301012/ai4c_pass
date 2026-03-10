import torch
import triton
import triton.language as tl

def pattern(in_2, sigmoid_out):
    tmp_4 = in_2 * sigmoid_out
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

def replacement_args(in_2, sigmoid_out):
    return (in_2, sigmoid_out)

@triton.jit
def fused_mul_hardtanh_kernel(
    x_ptr,           # in_2: [batch, channels, H, W]
    scale_ptr,       # sigmoid_out: [batch, channels, 1, 1] to be broadcast
    out_ptr,         # output: [batch, channels, H, W]
    batch_size,
    channels,
    height,
    width,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
):
    # Calculate program offsets (using 3 dimensions: batch, channel, spatial_x)
    pid_batch = tl.program_id(0)  # batch dimension
    pid_channel = tl.program_id(1)  # channel dimension
    pid_spatial = tl.program_id(2)  # spatial flattened dimension (x*y)

    # Check bounds
    if pid_batch >= batch_size or pid_channel >= channels:
        return
    
    # Unflatten spatial dimension
    spatial_per_channel = height * width
    local_pid = pid_spatial
    
    # Flatten spatial index to x,y coordinates
    pid_y = local_pid // width
    pid_x = local_pid % width
    
    if pid_x >= width or pid_y >= height:
        return

    # Load scale value (broadcast from [batch, channels, 1, 1])
    scale = tl.load(scale_ptr + pid_batch * channels + pid_channel)

    # Calculate element indices
    base_idx = (pid_batch * channels * height + pid_channel * height + pid_y) * width
    x_idx = base_idx + pid_x
    out_idx = base_idx + pid_x

    # Load input value
    x_val = tl.load(x_ptr + x_idx)

    # Multiply by scale
    mul_result = x_val * scale

    # Apply HardTanh: clamp between min_val and max_val
    result = tl.where(mul_result < min_val, min_val, 
                     tl.where(mul_result > max_val, max_val, mul_result))

    # Store result
    tl.store(out_ptr + out_idx, result)

@torch.fx.wrap
def fused_broadcast_mul_hardtanh(in_2, sigmoid_out):
    # Get tensor shapes
    batch_size, channels, height, width = in_2.shape
    
    # Create output tensor
    output = torch.empty((batch_size, channels, height, width), 
                        device=in_2.device, dtype=in_2.dtype)
    
    # Total spatial elements
    total_spatial = height * width
    
    # Use 3D grid: (batch_size, channels, total_spatial)
    grid = (batch_size, channels, total_spatial)
    
    # Launch kernel
    fused_mul_hardtanh_kernel[grid](
        x_ptr=in_2,
        scale_ptr=sigmoid_out,
        out_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        min_val=0.0,
        max_val=6.0,
    )
    
    return output

def replacement_func():
    return fused_broadcast_mul_hardtanh