import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern matching for Conv2D + Unfold fusion in float32 version"""
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_0 = None
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=(2, 2), stride=(2, 2))
    tmp_1 = None
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    tmp_2 = None
    return (tmp_3,)

def replacement_args(in_0, in_1):
    """Extract arguments for fused operation"""
    return (in_0, in_1)

@triton.jit
def fused_conv_unfold_kernel(
    x_ptr,                    # Input pointer: [1, 256, 32, 32]
    weight_ptr,              # Weight pointer: [128, 256, 1, 1]
    output_ptr,              # Output pointer: [1, 512, 16, 16]
    n_patches,               # Number of patches: 16*16 = 256
    in_channels,             # Input channels: 256
    out_channels_per_group,  # Output channels per group: 128
    in_height,               # Input height: 32
    in_width,                # Input width: 32
    patch_size: tl.constexpr,               # Patch size: 2
    stride: tl.constexpr,                   # Stride: 2
    block_size: tl.constexpr,               # Block size for parallelization
):
    pid = tl.program_id(0)
    
    # Calculate block of patches to process
    patches_per_block = (n_patches + block_size - 1) // block_size
    block_start = pid * patches_per_block
    block_end = min(block_start + patches_per_block, n_patches)
    
    for patch_idx in range(block_start, block_end):
        # Calculate patch position in output grid
        patch_height = patch_idx // 16
        patch_width = patch_idx % 16
        
        # For each output channel group (128 total, each with 4 channels)
        for oc_group in range(128):
            for oc_in_group in range(4):  # 4 channels per group
                oc_total = oc_group * 4 + oc_in_group
                
                # Process each patch position
                for ky in range(patch_size):
                    for kx in range(patch_size):
                        # Calculate input coordinates for this patch element
                        ih = patch_height * stride + ky
                        iw = patch_width * stride + kx
                        ih_orig = ih * 2  # Map back to original input before unfold
                        iw_orig = iw * 2
                        
                        # Calculate output offset
                        out_idx = patch_idx * 512 + oc_total
                        
                        # Perform fused convolution + extraction
                        # Load input: [B, C, H, W] -> [B, C, h, w]
                        if ih_orig < in_height and iw_orig < in_width:
                            input_val = tl.load(
                                x_ptr + ih_orig * in_width + iw_orig,
                                mask=(ih_orig < in_height) & (iw_orig < in_width),
                                other=0.0
                            )
                            
                            # Load weight and perform convolution (1x1)
                            weight_val = tl.load(weight_ptr + oc_total * in_channels)
                            
                            # Apply 1x1 convolution effectively
                            conv_result = input_val * weight_val
                            
                            # Store result
                            tl.store(output_ptr + out_idx, conv_result)

@triton.jit
def final_reshape_kernel(
    input_ptr,    # Input: [1, 512, 256] (flattened from [1, 512, 16, 16])
    output_ptr,   # Output: [1, 128, 4, 1024] (flattened)
    n_elements,   # Total elements: 512*256 = 131072
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    
    block_start = pid * block_size
    block_end = min(block_start + block_size, n_elements)
    
    for idx in range(block_start, block_end):
        # Reshape [1, 512, 16, 16] -> [1, 128, 4, 1024]
        # Input: idx = batch * (512*256) + c * 256 + spatial_idx
        # Output: idx_out = batch * (128*4*1024) + (c//4) * (4*1024) + (c%4) * 1024 + spatial_idx
        c_total = idx // 256  # Channel index (0-511)
        spatial_idx = idx % 256  # Spatial index (0-255)
        
        out_group = c_total // 4        # 0-127
        element_in_group = c_total % 4   # 0-3
        out_spatial = spatial_idx  # 16*16=256 spatial elements
        
        # Mapping to [1, 128, 4, 1024] output
        # Each group has 4 elements * 1024 spatial
        idx_out = out_group * 4096 + element_in_group * 1024 + out_spatial
        
        # Load input and store output
        input_val = tl.load(input_ptr + idx, mask=idx < n_elements, other=0.0)
        tl.store(output_ptr + idx_out, input_val)

@torch.fx.wrap
def fused_conv_unfold_operation_float32(weight_tensor, input_tensor):
    """Fused Conv2D + Unfold + Reshape operation for float32 pattern"""
    # Get input dimensions and metadata
    batch, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, _, _ = weight_tensor.shape
    
    # Calculate output dimensions for unfold operation
    patch_size = 2
    stride = 2
    out_height = in_height // stride  # 16
    out_width = in_width // stride    # 16
    n_patches = out_height * out_width    # 256
    out_channels_total = out_channels * patch_size * patch_size  # 128*4=512
    
    # Step 1: Fused Conv2D + Unfold
    intermediate = torch.empty((1, out_channels_total, n_patches), 
                              dtype=input_tensor.dtype, 
                              device=input_tensor.device)
    
    # Determine grid configuration
    block_size = 1024
    num_blocks = (n_patches + block_size - 1) // block_size
    
    # Launch fused kernel
    fused_conv_unfold_kernel[(num_blocks,)](
        x_ptr=input_tensor,
        weight_ptr=weight_tensor,
        output_ptr=intermediate,
        n_patches=n_patches,
        in_channels=in_channels,
        out_channels_per_group=out_channels,
        in_height=in_height,
        in_width=in_width,
        patch_size=patch_size,
        stride=stride,
        block_size=block_size
    )
    
    # Step 2: Final reshape to [1, 128, 4, 1024]
    total_elements = out_channels_total * n_patches  # 512*256 = 131072
    output = torch.empty((1, 128, 4, 1024),
                        dtype=input_tensor.dtype,
                        device=input_tensor.device)
    
    # Launch reshape kernel
    reshape_blocks = (total_elements + block_size - 1) // block_size
    final_reshape_kernel[(reshape_blocks,)](
        input_ptr=intermediate,
        output_ptr=output,
        n_elements=total_elements,
        block_size=block_size
    )
    
    return (output,)  # Return tuple to match pattern

def replacement_func():
    """Return the fused operation function"""
    return fused_conv_unfold_operation_float32