import torch
import triton
import triton.language as tl
import math

def pattern(in_1):
    tmp_0 = torch.nn.functional.unfold(in_1, kernel_size=(384, 384), stride=(192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    return tmp_2

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def fused_unfold_preshape_kernel(
    input_ptr,
    output_ptr,
    input_n,
    input_c, 
    input_h,
    input_w,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    output_patches_h,
    output_patches_w,
    output_total_patches,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    patch_idx = pid // (output_patches_h * output_patches_w)
    patch_local_idx = pid % (output_patches_h * output_patches_w)
    patch_y = patch_local_idx // output_patches_w
    patch_x = patch_local_idx % output_patches_w
    
    if patch_idx >= output_total_patches:
        return
    
    # Calculate starting position in input tensor for this patch
    start_y = patch_y * stride_h
    start_x = patch_x * stride_w
    
    # Calculate output block indices
    out_y = patch_idx // 3
    out_x = patch_idx % 3
    out_c = tl.arange(0, BLOCK_C)
    out_c = out_c.to(tl.int32)
    
    # Calculate input positions
    input_y = start_y + tl.arange(0, kernel_h)
    input_x = start_x + tl.arange(0, kernel_w)
    input_c = tl.arange(0, BLOCK_C)
    
    input_y = input_y.to(tl.int32)
    input_x = input_x.to(tl.int32)
    
    # Calculate output positions
    out_base_idx = out_y * 3 * kernel_h * kernel_w + out_x * kernel_h * kernel_w
    
    mask_c = out_c < input_c
    mask_y = input_y < input_h
    mask_x = input_x < input_w
    
    # For each channel in the block
    for c in range(0, input_c, BLOCK_C):
        channels = tl.arange(c, c + BLOCK_C).to(tl.int32)
        channels = channels.to(tl.int32)
        
        # Load data for this channel block
        data = tl.load(input_ptr + input_n * input_c * input_h * input_w + 
                      (channels[:, None, None] * input_h * input_w + 
                       input_y[None, :, None] * input_w + 
                       input_x[None, None, :]), 
                      mask=mask_c[:, None, None] & mask_y[None, :, None] & mask_x[None, None, :], 
                      other=0.0)
        
        # Store in output format: [patch_idx, channel, h, w]
        output_idx = (out_base_idx + channels * kernel_h * kernel_w)[None, :, None] + \
                    input_y[None, None, :] * kernel_w + \
                    input_x[None, None, :]
        
        output_base = patch_idx * 3 * kernel_h * kernel_w + \
                     out_base_idx + \
                     channels * kernel_h * kernel_w
        
        tl.store(output_ptr + output_base * BLOCK_SIZE + output_idx, data)

@torch.fx.wrap
def fused_unfold_preshape(input_tensor):
    input_n, input_c, input_h, input_w = input_tensor.shape
    kernel_h, kernel_w = 384, 384
    stride_h, stride_w = 192, 192
    
    # Calculate number of output patches
    output_patches_h = (input_h - kernel_h) // stride_h + 1
    output_patches_w = (input_w - kernel_w) // stride_w + 1
    output_total_patches = output_patches_h * output_patches_w
    
    # Calculate output shape
    output_total_elements = output_total_patches * 3 * kernel_h * kernel_w
    
    output = torch.empty((output_total_patches, 3, kernel_h, kernel_w), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    BLOCK_SIZE = 128
    BLOCK_C = min(3, 32)
    
    grid = (output_total_patches * 3 * kernel_h * kernel_w + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_unfold_preshape_kernel[grid](
        input_tensor,
        output,
        input_n,
        input_c,
        input_h, 
        input_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        output_patches_h,
        output_patches_w,
        output_total_patches,
        BLOCK_SIZE,
        BLOCK_C
    )
    
    return output

def replacement_func():
    return fused_unfold_preshape