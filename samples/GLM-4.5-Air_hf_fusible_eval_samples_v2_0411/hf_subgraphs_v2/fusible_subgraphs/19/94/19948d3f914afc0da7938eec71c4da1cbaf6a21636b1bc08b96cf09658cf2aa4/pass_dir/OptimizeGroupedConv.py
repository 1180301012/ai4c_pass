import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    conv2d = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 4)
    return conv2d

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def optimized_grouped_conv_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels_per_group,
    out_channels_per_group,
    in_height,
    in_width,
    out_height,
    out_width,
    kernel_h,
    kernel_w,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # 3D grid: batch, output_channel_group, spatial
    b = tl.program_id(0)
    oc_group = tl.program_id(1)
    hw_offset = tl.program_id(2) * (BLOCK_SIZE_H * BLOCK_SIZE_W)
    
    # Initialize pointers for current batch and output channel group
    input_base = input_ptr + b * in_channels_per_group * in_height * in_width
    output_base = output_ptr + b * out_channels_per_group * out_height * out_width + oc_group * BLOCK_SIZE_OC * out_height * out_width
    
    # Load bias for this output channel group
    bias_vals = tl.load(bias_ptr + oc_group * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC), mask=tl.arange(0, BLOCK_SIZE_OC) < out_channels_per_group, other=0.0)
    
    # Process spatial tiles
    for hi in range(0, out_height, BLOCK_SIZE_H):
        for wi in range(0, out_width, BLOCK_SIZE_W):
            h_idx = hi + tl.arange(0, BLOCK_SIZE_H)
            w_idx = wi + tl.arange(0, BLOCK_SIZE_W)
            
            # Initialize accumulator for this spatial tile
            acc = tl.zeros((BLOCK_SIZE_OC, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
            
            # Process input channel groups for this output channel group
            for ic in range(0, in_channels_per_group):
                # Load input: [1, in_channels_per_group, in_height, in_width] -> for current batch and input channel
                input_ptr_loc = input_base + ic * in_height * in_width + h_idx[:, None] * in_width + w_idx[None, :]
                input_vals = tl.load(input_ptr_loc, mask=h_idx[:, None] < out_height & w_idx[None, :] < out_width, other=0.0)
                
                # Load weights: [out_channels_per_group, in_channels_per_group_per_group, kernel_h, kernel_w]
                # For groups=4, each group handles in_channels_per_group=8, out_channels_per_group=8
                weight_ptr_loc = weight_ptr + oc_group * out_channels_per_group + tl.arange(0, BLOCK_SIZE_OC)[:, None] * in_channels_per_group + ic
                weight_vals = tl.load(weight_ptr_loc, mask=tl.arange(0, BLOCK_SIZE_OC)[:, None] < out_channels_per_group, other=0.0).to(tl.float32)
                
                # Accumulate: convolve 1x1 kernel
                acc += weight_vals[:, None, None] * input_vals[None, :, :]
            
            # Add bias and store result
            output_ptr_loc = output_base + (h_idx[:, None] * out_width + w_idx[None, :]) * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)[:, None, None]
            output_vals = acc + bias_vals[:, None, None]
            tl.store(output_ptr_loc, output_vals.to(tl.float16), mask=(tl.arange(0, BLOCK_SIZE_OC)[:, None, None] < out_channels_per_group) & (h_idx[:, None] < out_height) & (w_idx[None, :] < out_width))

@torch.fx.wrap
def optimized_grouped_conv(input_tensor, weight_tensor, bias_tensor):
    # Get tensor shapes
    batch_size, in_channels_total, in_height, in_width = input_tensor.shape
    out_channels_total, in_channels_per_group, kernel_h, kernel_w = weight_tensor.shape
    groups = 4  # Hardcoded based on the operation
    
    # Verify this is indeed a grouped conv with expected groups
    assert in_channels_total == groups * in_channels_per_group, f"Expected {groups * in_channels_per_group} input channels, got {in_channels_total}"
    assert out_channels_total == groups * in_channels_per_group, f"Expected {groups * in_channels_per_group} output channels, got {out_channels_total}"
    
    # For 1x1 convolution, output dimensions equal input dimensions
    out_height, out_width = in_height, in_width
    out_channels_per_group = in_channels_per_group  # Each group maintains same channel count
    
    # Determine optimal block sizes for GPU occupancy
    blocks_mixed_12 = 1024 // (BLOCK_SIZE_OC * BLOCK_SIZE_H * BLOCK_SIZE_W)
    blocks_mixed_12 = max(1, min(blocks_mixed_12, 256))
    
    BLOCK_SIZE_B = 1  # Process one batch at a time (batch_size=1 in our case)
    BLOCK_SIZE_OC = 32  # Output channels per group
    BLOCK_SIZE_IC = 8   # Input channels per group (matches our in_channels_per_group)
    BLOCK_SIZE_H = 8    # Spatial height block  
    BLOCK_SIZE_W = 8    # Spatial width block
    
    # Calculate grid size
    grid_b = (batch_size + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B
    grid_oc = (out_channels_per_group + BLOCK_SIZE_OC - 1) // BLOCK_SIZE_OC
    grid_hw = ((out_height * out_width) + (BLOCK_SIZE_H * BLOCK_SIZE_W) - 1) // (BLOCK_SIZE_H * BLOCK_SIZE_W)
    grid_size = (grid_b, grid_oc * groups, grid_hw)
    
    # Allocate output tensor
    output = torch.empty((batch_size, out_channels_total, out_height, out_width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    optimized_grouped_conv_kernel[grid_size](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        batch_size=batch_size,
        in_channels_per_group=in_channels_per_group,
        out_channels_per_group=out_channels_per_group,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_IC=BLOCK_SIZE_IC,
        BLOCK_SIZE_OC=BLOCK_SIZE_OC,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return output

def replacement_func():
    return optimized_grouped_conv