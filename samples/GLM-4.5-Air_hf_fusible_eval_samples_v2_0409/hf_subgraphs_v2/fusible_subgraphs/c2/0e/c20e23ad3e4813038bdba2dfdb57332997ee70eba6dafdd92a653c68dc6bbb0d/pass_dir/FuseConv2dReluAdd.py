import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv2d_relu_add_kernel(
    input_ptr, input_shape_ptr,
    weight_ptr, weight_shape_ptr,
    bias_ptr, bias_shape_ptr,
    residual_ptr, residual_shape_ptr,
    output_ptr, output_shape_ptr,
    batch_size, in_channels, out_channels,
    input_height, input_width,
    kernel_height, kernel_width,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    BLOCK_SIZE: tl.constexpr, BLOCK_WIDTH: tl.constexpr
):
    # Calculate output dimensions
    output_height = (input_height + 2 * pad_h - kernel_height * dilation_h) // stride_h + 1
    output_width = (input_width + 2 * pad_w - kernel_width * dilation_w) // stride_w + 1
    
    # Program indices
    pid = tl.program_id(0)
    n_elements = output_height * output_width
    
    # Calculate which spatial positions this program handles
    spatial_start = pid * BLOCK_SIZE
    spatial_ids = spatial_start + tl.arange(0, BLOCK_SIZE)
    mask = spatial_ids < n_elements
    
    # Convert spatial IDs to (h, w) coordinates
    h_ids = spatial_ids // output_width
    w_ids = spatial_ids % output_width
    
    # Process through output channels in blocks
    oc_block_start = tl.program_id(1) * BLOCK_WIDTH
    oc_block = oc_block_start + tl.arange(0, BLOCK_WIDTH)
    mask_oc = oc_block < out_channels
    
    # Initialize output for this output channel block
    oc_block_min = min(BLOCK_WIDTH, out_channels - oc_block_start)
    output_block = tl.zeros((oc_block_min, n_elements), dtype=tl.float16)
    
    # Convolution computation
    for oc_idx in range(oc_block_min):
        oc = oc_block[oc_idx]
        output_channel = tl.zeros(n_elements, dtype=tl.float16)
        
        # Iterate through input channels/groups
        for ic in range(groups):
            group_out_channels = out_channels // groups
            group_in_channels = in_channels // groups
            
            # Weight pointer for this group and output channel
            weight_offset = (oc // group_out_channels) * group_out_channels * group_in_channels * kernel_height * kernel_width + \
                           (oc % group_out_channels) * group_in_channels * kernel_height * kernel_width
            weight_group_ptr = weight_ptr + weight_offset
            
            # Input pointer for this group
            input_offset = (ic // group_in_channels) * group_in_channels * input_height * input_width
            input_group_ptr = input_ptr + input_offset
            
            # Convolution for this input/output channel pair
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input coordinates with padding and dilation
                    ih_base = h_ids * stride_h - pad_h + kh * dilation_h
                    iw_base = w_ids * stride_w - pad_w + kw * dilation_w
                    
                    # Create mask for valid input positions
                    ih_mask = (ih_base >= 0) & (ih_base < input_height)
                    iw_mask = (iw_base >= 0) & (iw_base < input_width)
                    valid_mask = mask & ih_mask & iw_mask
                    
                    # Weight for this kernel position
                    weight_idx = ic * kernel_height * kernel_width + kh * kernel_width + kw
                    weight_value = tl.load(weight_group_ptr + weight_idx, other=0.0)
                    
                    # Load input values
                    ih_ids = ih_base * ih_mask + (~ih_mask) * 0
                    iw_ids = iw_base * iw_mask + (~iw_mask) * 0
                    input_idx = ih_ids * input_width + iw_ids
                    
                    input_values = tl.load(input_group_ptr + input_idx, mask=valid_mask, other=0.0)
                    
                    # Accumulate convolution
                    acc = tl.zeros(n_elements, dtype=tl.float16)
                    tl.atomic_add(acc, input_values * weight_value, mask=valid_mask)
                    output_channel += acc
            
            # Add bias
            bias_value = tl.load(bias_ptr + oc, other=0.0)
            output_channel += bias_value
        
        # Store intermediate result
        tl.store(output_block + oc_idx * n_elements, output_channel, mask=mask)
    
    # Apply ReLU
    output_block = tl.maximum(output_block, 0.0)
    
    # Load residual and add
    residual_block = tl.load(
        residual_ptr + oc_block_start * output_height * output_width, 
        mask=mask_oc[:, None], 
        other=0.0
    )
    
    # Final output: conv_out + residual
    final_output = output_block + residual_block
    
    # Store final output
    tl.store(
        output_ptr + oc_block_start * output_height * output_width,
        final_output,
        mask=mask_oc[:, None]
    )

@torch.fx.wrap
def fused_conv2d_relu_add_op(
    input_tensor, weight_tensor, bias_tensor, 
    residual_tensor, output_shape
):
    # Get tensor shapes
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight_tensor.shape
    
    # Output dimensions (should match residual in this case)
    output_height, output_width = output_shape[-2:]
    
    # Calculate grid size
    BLOCK_SIZE = 1024
    BLOCK_WIDTH = 64
    
    n_spatial_elements = output_height * output_width
    n_spatial_blocks = (n_spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_channel_blocks = (out_channels + BLOCK_WIDTH - 1) // BLOCK_WIDTH
    
    # Create output tensor
    output_tensor = torch.empty(output_shape, dtype=torch.float16, device=input_tensor.device)
    
    # Launch kernel
    fused_conv2d_relu_add_kernel[(n_spatial_blocks, n_channel_blocks)](
        input_tensor, input_tensor.shape,
        weight_tensor, weight_tensor.shape,
        bias_tensor, bias_tensor.shape,
        residual_tensor, residual_tensor.shape,
        output_tensor, output_shape,
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_height, kernel_width,
        2, 2,  # stride_h, stride_w
        1, 1,  # pad_h, pad_w
        1, 1,  # dilation_h, dilation_w
        1,     # groups
        BLOCK_SIZE, BLOCK_WIDTH
    )
    
    return output_tensor

def pattern(in_3, in_1, in_0, in_2):
    conv2d = torch.conv2d(in_3, in_1, in_0, (2, 2), (1, 1), (1, 1), 1)
    tmp_3 = torch.nn.functional.relu(conv2d, inplace=True)
    tmp_4 = in_2 + tmp_3
    return tmp_4

def replacement_args(in_3, in_1, in_0, in_2):
    return (in_3, in_1, in_0, in_2)

def replacement_func():
    return fused_conv2d_relu_add_op