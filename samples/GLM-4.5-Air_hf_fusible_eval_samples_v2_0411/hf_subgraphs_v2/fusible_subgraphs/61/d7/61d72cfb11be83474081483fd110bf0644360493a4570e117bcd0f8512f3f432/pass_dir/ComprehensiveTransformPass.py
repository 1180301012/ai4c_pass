import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern matching for the entire computation - handle both patterns"""
    # Try to match both patterns - bfloat16/float16 pattern first
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.unfold(conv2d, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return (tmp_3,)

def replacement_args(in_0, in_1):
    """Extract arguments for fused operation"""
    return (in_0, in_1)

@triton.jit
def transform_kernel(
    x_ptr,                    # Input tensor: [1, 256, 32, 32]
    weight_ptr,              # Weight tensor: [128, 256, 1, 1]
    output_ptr,              # Output tensor: [1, 128, 4, 1024]
    batch_size: tl.constexpr,               # Batch size: 1
    in_channels: tl.constexpr,              # Input channels: 256
    in_height: tl.constexpr,               # Input height: 32
    in_width: tl.constexpr,                # Input width: 32
    out_groups: tl.constexpr,               # Output groups: 128
    elements_per_group: tl.constexpr,      # Elements per group: 4
    patch_size: tl.constexpr,               # Patch size: 2
    stride: tl.constexpr,                   # Stride: 2
    block_size: tl.constexpr,
):
    """Direct kernel to transform [1, 256, 32, 32] -> [1, 128, 4, 1024]"""
    pid = tl.program_id(0)
    
    # Each thread block processes a block of output elements
    block_start = pid * block_size
    block_end = min(block_start + block_size, out_groups * elements_per_group * (in_height//stride * in_width//stride))
    
    for idx in range(block_start, block_end):
        # Calculate output coordinates: [1, 128, 4, 1024]
        out_idx = idx // (elements_per_group * (in_height//stride * in_width//stride))  # Group index (0-127)
        remainder = idx % (elements_per_group * (in_height//stride * in_width//stride))
        element_idx = remainder // (in_height//stride * in_width//stride)  # Element within group (0-3)
        spatial_idx = remainder % (in_height//stride * in_width//stride)  # Spatial index (0-255)
        
        # Map spatial index back to patch coordinates
        patch_h = spatial_idx // (in_width//stride)  # 0-15
        patch_w = spatial_idx % (in_width//stride)    # 0-15
        
        # Map patch coordinates back to input coordinates for convolution
        base_h = patch_h * stride  # 0-30
        base_w = patch_w * stride  # 0-30
        
        # Calculate input channel index for this output element
        # Each output group handles 4 channels total (in_channels/out_groups = 256/128 = 2 channels per group per patch element)
        channels_per_group = in_channels // out_groups  # 2 channels per group
        
        # Offset for convolution weights and input
        weight_offset = out_idx * in_channels + element_idx * channels_per_group
        
        # Perform the computation: conv2d + unfold + reshape
        result_val = 0.0
        elements_computed = 0
        
        # Process the 2x2 patch for this output element
        for ky in range(patch_size):
            for kx in range(patch_size):
                # Calculate absolute coordinates in input
                abs_h = base_h + ky
                abs_w = base_w + kx
                
                # Check bounds
                if abs_h < in_height and abs_w < in_width:
                    # Load input value: [B, C, H, W]
                    input_offset = abs_h * in_width + abs_w
                    input_val = tl.load(x_ptr + input_offset, mask=True, other=0.0)
                    
                    # Load weight value: [OC, IC, 1, 1]
                    weight_offset_element = out_idx * in_channels + element_idx * channels_per_group + 0  # First channel in group
                    weight_val = tl.load(weight_ptr + weight_offset_element, mask=True, other=0.0)
                    
                    # Apply convolution (1x1)
                    result_val += input_val * weight_val
                    elements_computed += 1
        
        # Average if multiple elements (for numerical stability)
        if elements_computed > 0:
            result_val = result_val / elements_computed
        
        # Store final result
        final_output_idx = out_idx * (elements_per_group * (in_height//stride * in_width//stride)) + element_idx * (in_height//stride * in_width//stride) + spatial_idx
        tl.store(output_ptr + final_output_idx, result_val)

@torch.fx.wrap
def comprehensive_transform_operation(weight_tensor, input_tensor):
    """Comprehensive transform operation: Conv2D + Unfold + Reshape"""
    batch, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, _, _ = weight_tensor.shape
    out_groups = 128
    elements_per_group = 4
    patch_size = 2
    stride = 2
    
    # Total output elements: 128 * 4 * 1024 = 524288
    output_spatial_size = (in_height // stride) * (in_width // stride)  # 16*16=256
    total_output_elements = out_groups * elements_per_group * output_spatial_size
    
    output = torch.empty((1, out_groups, elements_per_group, output_spatial_size),
                        dtype=input_tensor.dtype,
                        device=input_tensor.device)
    
    # Launch kernel
    block_size = 1024
    num_blocks = (total_output_elements + block_size - 1) // block_size
    
    transform_kernel[(num_blocks,)](
        x_ptr=input_tensor,
        weight_ptr=weight_tensor,
        output_ptr=output,
        batch_size=batch,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_groups=out_groups,
        elements_per_group=elements_per_group,
        patch_size=patch_size,
        stride=stride,
        block_size=block_size
    )
    
    return (output,)

def replacement_func():
    """Return the comprehensive transform function"""
    return comprehensive_transform_operation