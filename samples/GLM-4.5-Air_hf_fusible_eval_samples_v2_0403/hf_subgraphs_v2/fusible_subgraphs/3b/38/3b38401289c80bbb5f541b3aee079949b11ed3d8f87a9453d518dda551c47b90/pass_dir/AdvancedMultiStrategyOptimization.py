import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    # Branch 1: in_1 processing (multiply + add)
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    
    # Branch 2: in_0 channel 1 processing (extract + unsqueeze + multiply + add)
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    
    # Branch 3: in_0 channel 2 processing (extract + unsqueeze + multiply + add)
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    
    # Final concatenation
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    
    return tmp_11

def replacement_args(in_1, in_0):
    return (in_1, in_0)

# Advanced Triton kernel with multiple optimization strategies
@triton.jit
def advanced_optimization_kernel(
    # Input tensors
    in_1, in_0, out,
    # Tensor metadata
    channels, height, width, spatial_size,
    # Scaling and bias parameters
    scale1, bias1, scale2, bias2, scale3, bias3,
    # Block sizes - using fixed sizes for arange compatibility
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # 2D grid layout for better GPU occupancy
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    # Fixed block sizes for compile-time constant arange
    block_x = BLOCK_SIZE_X
    block_y = BLOCK_SIZE_Y
    
    # Calculate linear memory offsets with 2D grid
    # This is a simpler approach that avoids unsupported operations
    linear_offset = pid_y * BLOCK_SIZE_Y * width + pid_x * BLOCK_SIZE_X
    offsets = linear_offset + tl.arange(0, BLOCK_SIZE_X * BLOCK_SIZE_Y)
    mask = offsets < spatial_size
    
    # Vectorized computation for all three branches with fused arithmetic
    # Branch 1: Process in_1 directly (multiply + add)
    in_1_vals = tl.load(in_1 + offsets, mask=mask, other=0.0)
    branch1_out = in_1_vals * scale1 + bias1
    
    # Branch 2: Process channel 1 from in_0 (multiply + add)
    channel1_offsets = offsets + 1 * spatial_size
    in_0_1_vals = tl.load(in_0 + channel1_offsets, mask=mask, other=0.0)
    branch2_out = in_0_1_vals * scale2 + bias2
    
    # Branch 3: Process channel 2 from in_0 (multiply + add)
    channel2_offsets = offsets + 2 * spatial_size
    in_0_2_vals = tl.load(in_0 + channel2_offsets, mask=mask, other=0.0)
    branch3_out = in_0_2_vals * scale3 + bias3
    
    # Store results to output in proper concatenation order
    # Output layout: [channel0, channel1, channel2] each of size [height, width]
    tl.store(out + offsets, branch1_out, mask=mask)
    tl.store(out + offsets + 1 * spatial_size, branch2_out, mask=mask)
    tl.store(out + offsets + 2 * spatial_size, branch3_out, mask=mask)

@torch.fx.wrap
def advanced_optimization(in_1, in_0):
    # Get tensor shapes and metadata
    batch_size, _, height, width = in_1.shape
    _, channels, _, _ = in_0.shape
    
    # Scale and bias parameters
    scale1, bias1 = 0.458, -0.030000000000000027
    scale2, bias2 = 0.448, -0.08799999999999997
    scale3, bias3 = 0.45, -0.18799999999999994
    
    # Spatial size for memory layout calculations
    spatial_size = height * width
    
    # Create output tensor: [batch, 3, height, width]
    output_shape = (batch_size, 3, height, width)
    out = torch.empty(output_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Process each batch with optimized scheduling
    for b in range(batch_size):
        # Extract contiguous tensors for current batch
        in_1_batch = in_1[b, 0].contiguous()
        in_0_batch = in_0[b].contiguous()
        out_batch = out[b].contiguous()
        
        # Choose optimal block sizes based on tensor dimensions
        BLOCK_SIZE_X = 64 if width >= 64 else 32
        BLOCK_SIZE_Y = 16 if height >= 64 else 8
        
        # Calculate grid dimensions for 2D parallelization
        grid_x = (width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
        grid_y = (height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
        
        # Launch optimized kernel
        advanced_optimization_kernel[grid_x, grid_y](
            in_1=in_1_batch,
            in_0=in_0_batch,
            out=out_batch,
            channels=channels,
            height=height,
            width=width,
            spatial_size=spatial_size,
            scale1=scale1, bias1=bias1,
            scale2=scale2, bias2=bias2,
            scale3=scale3, bias3=bias3,
            BLOCK_SIZE_X=BLOCK_SIZE_X,
            BLOCK_SIZE_Y=BLOCK_SIZE_Y,
        )
    
    return out

def replacement_func():
    return advanced_optimization