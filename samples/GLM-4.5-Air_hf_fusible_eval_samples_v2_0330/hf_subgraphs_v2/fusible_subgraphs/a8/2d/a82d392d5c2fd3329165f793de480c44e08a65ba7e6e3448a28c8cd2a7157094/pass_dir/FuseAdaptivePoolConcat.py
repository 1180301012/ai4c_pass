import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Fuse adaptive_avg_pool2d + concat pattern"""
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def adaptive_pool2d_concat_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    batch_size,
    in_0_channels,
    in_1_channels,
    in_0_height,
    in_0_width,
    out_height,
    out_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Calculate output position
    m = pid_m * BLOCK_SIZE_M
    n = pid_n * BLOCK_SIZE_N
    
    # Simple adaptive pooling: average over 2x2 or appropriate block
    stride_y = in_0_height // out_height
    stride_x = in_0_width // out_width
    
    # Process each output pixel in the block
    for i in range(BLOCK_SIZE_M):
        for j in range(BLOCK_SIZE_N):
            out_y = m + i
            out_x = n + j
            
            if out_y < out_height and out_x < out_width:
                # Define input region bounds
                in_y_start = out_y * stride_y
                in_y_end = min(in_y_start + stride_y, in_0_height)
                in_x_start = out_x * stride_x
                in_x_end = min(in_x_start + stride_x, in_0_width)
                
                # Average pooling
                sum_val = 0.0
                count = 0
                
                for in_y in range(in_y_start, in_y_end):
                    # Load and average all channels for this position
                    channel_sum = 0.0
                    
                    for in_x in range(in_x_start, in_x_end):
                        offset = pid_b * in_0_channels * in_0_height * in_0_width + \
                                in_y * in_0_width * in_0_channels + \
                                in_x * in_0_channels
                        val = tl.load(in_0_ptr + offset, mask=True, other=0.0).to(tl.float32)
                        channel_sum += val
                    
                    if (in_x_end - in_x_start) > 0:
                        channel_sum /= (in_x_end - in_x_start)
                    sum_val += channel_sum
                    count += 1
                
                if count > 0:
                    avg_val = sum_val / count
                else:
                    avg_val = 0.0
                
                # Store averaged pooled values for all channels
                output_base = pid_b * (in_0_channels + in_1_channels) * out_height * out_width + \
                            out_y * (in_0_channels + in_1_channels) * out_width + \
                            out_x * (in_0_channels + in_1_channels)
                
                for c in range(in_0_channels):
                    tl.store(out_ptr + output_base + c, avg_val)
                
                # Store in_1 values for all channels
                in_1_base = pid_b * in_1_channels * out_height * out_width + \
                          out_y * in_1_channels * out_width + \
                          out_x * in_1_channels
                
                for c in range(in_1_channels):
                    src_offset = output_base + in_0_channels + c
                    val = tl.load(in_1_ptr + in_1_base + c)
                    tl.store(out_ptr + src_offset, val)

@torch.fx.wrap
def fused_adaptive_pool_concat(in_0, in_1):
    # Get input shapes
    batch_size = in_0.shape[0]
    in_0_channels = in_0.shape[1]
    in_1_channels = in_1.shape[1]
    in_0_height = in_0.shape[2]
    in_0_width = in_0.shape[3]
    out_height = 32
    out_width = 24
    
    # Output shape
    out_shape = (batch_size, in_0_channels + in_1_channels, out_height, out_width)
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Determine grid configuration - use larger blocks for better occupancy
    BLOCK_SIZE_M = 4
    BLOCK_SIZE_N = 4
    grid_m = (out_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n, batch_size)
    
    # Launch kernel
    adaptive_pool2d_concat_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        in_0_channels=in_0_channels,
        in_1_channels=in_1_channels,
        in_0_height=in_0_height,
        in_0_width=in_0_width,
        out_height=out_height,
        out_width=out_width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return fused_adaptive_pool_concat