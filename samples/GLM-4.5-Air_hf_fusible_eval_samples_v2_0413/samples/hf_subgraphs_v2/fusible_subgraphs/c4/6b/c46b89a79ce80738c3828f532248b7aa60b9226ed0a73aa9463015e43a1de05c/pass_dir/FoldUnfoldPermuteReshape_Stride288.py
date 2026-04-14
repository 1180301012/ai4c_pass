import torch
import triton
import triton.language as tl

def pattern(input_tensor, kernel_size, stride):
    # Match: unfold -> permute(2, 0, 1) -> reshape(-1, 3, 384, 384)
    tmp_0 = torch.nn.functional.unfold(input_tensor, kernel_size=kernel_size, stride=stride)
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    return tmp_2

def replacement_args(input_tensor, kernel_size, stride):
    return (input_tensor, kernel_size, stride)

@triton.jit
def unfold_reshape_kernel(
    input_ptr,
    output_ptr,
    input_stride_batch,
    input_stride_channels_in,
    input_stride_height_in,
    input_stride_width_in,
    input_channels_in,
    input_height_in, 
    input_width_in,
    output_stride_patches,
    output_stride_batch,
    output_stride_channels,
    output_stride_height,
    output_stride_width,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    num_patches_dim0,
    patches_dim0,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    program_id = tl.program_id(0)
    
    # Calculate which patch we're processing
    patch_id_base = program_id * BLOCK_SIZE_M
    patch_id = patch_id_base + tl.arange(0, BLOCK_SIZE_M)[:, None]
    
    # Calculate the output positions for each patch
    batch_idx = patch_id // patches_dim0
    patch_idx = patch_id % patches_dim0
    
    # Calculate which input tile this patch corresponds to
    patch_row = patch_idx // num_patches_dim0
    patch_col = patch_idx % num_patches_dim0
    input_start_row = patch_row * stride_height
    input_start_col = patch_col * stride_width
    
    # Calculate total pixels in a patch
    patch_height = kernel_height
    patch_width = kernel_width
    patch_pixels = patch_height * patch_width
    
    # Process channels and pixels within each patch
    channel_off = tl.arange(0, BLOCK_SIZE_N)[None, :]
    pixel_off = tl.arange(0, BLOCK_SIZE_N)[None, :] // patch_width
    pixel_off_x = tl.arange(0, BLOCK_SIZE_N) % patch_width
    
    # Create mask for valid computations
    mask_channel = channel_off < 3
    mask_pixel = pixel_off < patch_height
    mask_full = mask_channel & mask_pixel
    
    # Calculate input positions
    input_batch_ptr = input_ptr + batch_idx * input_stride_batch
    input_row_base = input_start_row + pixel_off
    input_col_base = input_start_col + pixel_off_x 
    
    input_channels_idx = channel_off
    input_ptr_pos = (input_batch_ptr + 
                    input_channels_idx * input_stride_channels_in + 
                    input_row_base * input_stride_height_in + 
                    input_col_base * input_stride_width_in)
    
    # Calculate output positions  
    batch_out = batch_idx * output_stride_batch
    patch_out = patch_idx * output_stride_patches
    
    channels_out_idx = (pixel_off * patch_width + pixel_off_x) // patch_pixels * 3 + channel_off
    height_out_idx = (pixel_off * patch_width + pixel_off_x) // patch_width
    width_out_idx = (pixel_off * patch_width + pixel_off_x) % patch_width
    
    output_ptr_pos = (output_ptr + 
                     batch_out * output_stride_batch + 
                     patch_out * output_stride_patches + 
                     channels_out_idx * output_stride_channels + 
                     height_out_idx * output_stride_height + 
                     width_out_idx * output_stride_width)
    
    # Load input data
    input_vals = tl.load(input_ptr_pos, mask=mask_full, other=0.0)
    
    # Store output data
    tl.store(output_ptr_pos, input_vals, mask=mask_full)

@torch.fx.wrap  
def fused_unfold_reshape(input_tensor, kernel_size, stride):
    _, channels_in, height_in, width_in = input_tensor.shape
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = stride
    
    # Calculate number of patches in each dimension
    num_patches_dim0 = (height_in - kernel_height) // stride_height + 1
    num_patches_dim1 = (width_in - kernel_width) // stride_width + 1
    total_patches = num_patches_dim0 * num_patches_dim1
    
    # Output shape: [total_patches, channels, kernel_height, kernel_width]
    output_shape = [total_patches, 3, kernel_height, kernel_width]
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set up kernel config
    BLOCK_SIZE_M = 32  # Number of patches to process per program
    BLOCK_SIZE_N = 128  # Number of elements to process per patch
    
    # Calculate grid size
    num_patches = total_patches
    grid_size = (triton.cdiv(num_patches, BLOCK_SIZE_M),)
    
    # Calculate input strides
    input_stride_batch = input_tensor.stride(0)
    input_stride_channels_in = input_tensor.stride(1) 
    input_stride_height_in = input_tensor.stride(2)
    input_stride_width_in = input_tensor.stride(3)
    
    # Calculate output strides
    output_stride_batch = output.stride(0)
    output_stride_patches = output.stride(1)
    output_stride_channels = output.stride(2)
    output_stride_height = output.stride(3) 
    output_stride_width = output.stride(4)
    
    # Launch kernel
    unfold_reshape_kernel[grid_size](
        input_ptr=input_tensor,
        output_ptr=output,
        input_stride_batch=input_stride_batch,
        input_stride_channels_in=input_stride_channels_in,
        input_stride_height_in=input_stride_height_in, 
        input_stride_width_in=input_stride_width_in,
        input_channels_in=channels_in,
        input_height_in=height_in,
        input_width_in=width_in,
        output_stride_patches=output_stride_patches,
        output_stride_batch=output_stride_batch,
        output_stride_channels=output_stride_channels,
        output_stride_height=output_stride_height,
        output_stride_width=output_stride_width,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        stride_height=stride_height,
        stride_width=stride_width,
        num_patches_dim0=num_patches_dim0,
        patches_dim0=total_patches,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    return fused_unfold_reshape