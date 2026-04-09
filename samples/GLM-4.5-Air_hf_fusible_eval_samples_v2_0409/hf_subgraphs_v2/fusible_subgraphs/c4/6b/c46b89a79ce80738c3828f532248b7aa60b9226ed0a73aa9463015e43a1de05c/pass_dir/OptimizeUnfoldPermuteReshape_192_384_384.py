import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    tmp_0 = torch.nn.functional.unfold(input_tensor, kernel_size = (384, 384), stride = (192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    return tmp_2

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_unfold_kernel_192(
    input_ptr,
    output_ptr,
    input_batch,
    input_channels,
    input_height,
    input_width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    output_patches,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate which patch this program handles
    patch_idx = pid
    
    if patch_idx >= output_patches:
        return
    
    # Calculate the spatial position for this patch
    output_h = output_height = (input_height - kernel_h) // stride_h + 1
    output_w = (input_width - kernel_w) // stride_w + 1
    
    h_pos = patch_idx // output_w
    w_pos = patch_idx % output_w
    
    # Calculate input offset for this patch
    input_h = h_pos * stride_h
    input_w = w_pos * stride_w
    
    # Process each channel in the patch
    for c in range(3):  # We only need 3 channels
        for block_offset in range(0, kernel_h * kernel_w, BLOCK_SIZE):
            remaining = min(BLOCK_SIZE, kernel_h * kernel_w - block_offset)
            offsets = block_offset + tl.arange(0, remaining)
            mask = offsets < kernel_h * kernel_w
            
            # Calculate input indices for this block of patch elements
            kw = offsets % kernel_w
            kh = offsets // kernel_w
            
            input_idx = ((c * input_height + input_h + kh) * input_width + input_w + kw).to(tl.int64)
            output_idx = (patch_idx * 3 + c) * kernel_h * kernel_w + offsets
            
            # Load from input and store to output
            input_data = tl.load(input_ptr + input_idx, mask=(input_idx < input_batch * input_channels * input_height * input_width), other=0.0)
            tl.store(output_ptr + output_idx, input_data, mask=mask)

@torch.fx.wrap
def optimized_unfold_192(input_tensor):
    input_shape = input_tensor.shape
    if len(input_shape) != 4:
        raise ValueError("Input must be 4D tensor")
    
    batch, channels, height, width = input_shape
    kernel_h, kernel_w = 384, 384
    stride_h, stride_w = 192, 192
    
    # Calculate output dimensions
    output_height = (height - kernel_h) // stride_h + 1
    output_width = (width - kernel_w) // stride_w + 1
    output_patches = output_height * output_width
    
    # Output should be [output_patches, 3, kernel_h, kernel_w]
    # Create output tensor
    output = torch.empty((output_patches, 3, kernel_h, kernel_w), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    num_programs = (output_patches + 127) // 128  # Use 128 patches per program
    BLOCK_SIZE = 1024  # Process 1024 elements per program
    
    grid = (num_programs,)
    
    optimized_unfold_kernel_192[grid if triton.testing.IS_NEW_TRITON else ()](
        input_tensor,
        output,
        batch,
        channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        output_patches,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_unfold_192