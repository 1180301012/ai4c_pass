import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor, scale_tensor):
    conv2d = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = scale_tensor * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(input_tensor, weight_tensor, bias_tensor, scale_tensor):
    return (input_tensor, weight_tensor, bias_tensor, scale_tensor)

@triton.jit
def fused_complete_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    out_ptr,
    batch_size,
    in_channels_total,
    out_channels_total,
    in_channels_per_group,
    out_channels_per_group,
    in_height,
    in_width,
    conv_out_height,
    conv_out_width,
    scale_height,
    scale_width,
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
    
    # Initialize pointers for current batch
    input_base = input_ptr + b * in_channels_total * in_height * in_width
    scale_base = scale_ptr + b * out_channels_total * scale_height * scale_width
    output_base = out_ptr + b * out_channels_total * scale_height * scale_width + oc_group * BLOCK_SIZE_OC * scale_height * scale_width
    
    # Process spatial tiles (using scale dimensions)
    for hi in range(0, scale_height, BLOCK_SIZE_H):
        for wi in range(0, scale_width, BLOCK_SIZE_W):
            h_idx = hi + tl.arange(0, BLOCK_SIZE_H)
            w_idx = wi + tl.arange(0, BLOCK_SIZE_W)
            
            # Initialize accumulator for this spatial tile (conv operates on 1x1 spatial)
            acc = tl.zeros((BLOCK_SIZE_OC, 1, 1), dtype=tl.float32)
            
            # Process input channel groups for this output channel group
            for ic_group in range(4):  # groups=4
                ic_base = ic_group * in_channels_per_group
                
                for ic in range(0, in_channels_per_group):
                    # Load input: [batch, in_channels_total, in_height, in_width] - spatial dims are 1x1
                    # Since conv input is [1, 32, 1, 1], all spatial positions are the same
                    input_ptr_loc = input_base + (ic_base + ic) * in_height * in_width
                    input_val = tl.load(input_ptr_loc)  # Load single value (position is always valid)
                    
                    # Load weights: For this input channel, load all output channels in the current output group
                    weight_offset = (oc_group * BLOCK_SIZE_OC)
                    weight_vals = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, BLOCK_SIZE_OC) < out_channels_per_group, other=0.0).to(tl.float32)
                    
                    # Convolution: 1x1 kernel - accumulate weighted input
                    acc += weight_vals[:, None, None] * input_val
            
            # Add bias (to all spatial positions)
            bias_vals = tl.load(bias_ptr + oc_group * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC), mask=tl.arange(0, BLOCK_SIZE_OC) < out_channels_per_group, other=0.0)
            acc = acc + bias_vals[:, None, None]
            
            # Apply sigmoid
            sigmoid_vals = tl.sigmoid(acc)
            
            # Broadcast to all spatial positions and apply channel-wise scaling
            for sp_h in range(h_idx.min(), h_idx.max() + 1):
                for sp_w in range(w_idx.min(), w_idx.max() + 1):
                    if sp_h < scale_height and sp_w < scale_width:
                        # Load scale tensor for this spatial position
                        scale_idx = (sp_h * scale_width + sp_w) * out_channels_total + oc_group * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)
                        scale_vals = tl.load(scale_ptr + scale_idx, mask=tl.arange(0, BLOCK_SIZE_OC) < out_channels_per_group, other=0.0)
                        
                        # Apply channel-wise scaling
                        output_vals = scale_vals * sigmoid_vals[:, 0, 0]
                        
                        # Store result directly to output 
                        output_idx = (sp_h * scale_width + sp_w) * out_channels_total + oc_group * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)
                        tl.store(output_ptr + output_idx, output_vals.to(tl.float16), mask=tl.arange(0, BLOCK_SIZE_OC) < out_channels_per_group)
    
    

@torch.fx.wrap
def fused_complete_convolution(input_tensor, weight_tensor, bias_tensor, scale_tensor):
    # Get tensor shapes
    batch_size, in_channels_total, in_height, in_width = input_tensor.shape
    out_channels_total, in_channels_per_group, kernel_h, kernel_w = weight_tensor.shape
    groups = 4  # groups=4 from the original operation
    
    # Verify shapes match expected grouped convolution
    assert in_channels_total == groups * in_channels_per_group, f"Expected {groups * in_channels_per_group} input channels, got {in_channels_total}"
    # For grouped conv with groups=4, output channels per group = out_channels_total / groups
    out_channels_per_group = out_channels_total // groups
    # The scale tensor (in_2) can have different spatial dimensions than the conv input
    scale_height, scale_width = scale_tensor.shape[2], scale_tensor.shape[3]
    assert scale_tensor.shape == (batch_size, out_channels_total, scale_height, scale_width), f"Scale tensor shape mismatch, expected {(batch_size, out_channels_total, scale_height, scale_width)}, got {scale_tensor.shape}"
    
    # For 1x1 convolution, conv output dimensions equal conv input dimensions  
    conv_out_height, conv_out_width = in_height, in_width
    
    # Optimize block sizes for GPU occupancy
    BLOCK_SIZE_B = 1  # Process one batch at a time
    BLOCK_SIZE_OC = 32  # Output channels per group
    BLOCK_SIZE_IC = 8   # Input channels per group
    BLOCK_SIZE_H = 8    # Spatial height block  
    BLOCK_SIZE_W = 8    # Spatial width block
    
    # Calculate grid size
    grid_b = (batch_size + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B
    grid_oc = (out_channels_per_group + BLOCK_SIZE_OC - 1) // BLOCK_SIZE_OC
    grid_hw = ((scale_height * scale_width) + (BLOCK_SIZE_H * BLOCK_SIZE_W) - 1) // (BLOCK_SIZE_H * BLOCK_SIZE_W)
    grid_size = (grid_b, grid_oc * groups, grid_hw)
    
    # Allocate output tensor (contiguous by construction) - use scale spatial dimensions
    output = torch.empty((batch_size, out_channels_total, scale_height, scale_width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch fused kernel
    fused_complete_kernel[grid_size](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        scale_ptr=scale_tensor,
        out_ptr=output,
        batch_size=batch_size,
        in_channels_total=in_channels_total,
        out_channels_total=out_channels_total,
        in_channels_per_group=in_channels_per_group,
        out_channels_per_group=out_channels_per_group,
        in_height=in_height,
        in_width=in_width,
        conv_out_height=conv_out_height,
        conv_out_width=conv_out_width,
        scale_height=scale_height,
        scale_width=scale_width,
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
    return fused_complete_convolution