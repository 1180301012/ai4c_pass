import torch
import triton
import triton.language as tl

def pattern(in_5, in_7, in_8, in_6, tmp_7):
    # Concatenation along dimension 1
    tmp_8 = torch.cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
    return tmp_8

def replacement_args(in_5, in_7, in_8, in_6, tmp_7):
    return (in_5, in_7, in_8, in_6, tmp_7)

@triton.jit
def concat_features_dim1_kernel(
    ptr0, ptr1, ptr2, ptr3, ptr4,  # Input tensor pointers
    out_ptr,
    channels0: tl.constexpr,
    channels1: tl.constexpr,
    channels2: tl.constexpr,
    channels3: tl.constexpr,
    channels4: tl.constexpr,
    batch: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate total channels
    total_channels = channels0 + channels1 + channels2 + channels3 + channels4
    
    # Each program handles one spatial position in the batch
    spatial_idx = tl.program_id(0)
    
    if spatial_idx >= batch * height * width:
        return
    
    # Calculate coordinates
    b = spatial_idx // (height * width)
    h = (spatial_idx % (height * width)) // width
    w = spatial_idx % width
    
    # Calculate output offset
    out_offset = ((b * total_channels + 0) * height + h) * width + w
    
    # Copy data from input tensor 0 (channels0)
    in_offset0 = ((b * channels0 + 0) * height + h) * width + w
    tile0 = tl.load(ptr0 + in_offset0)
    tl.store(out_ptr + out_offset, tile0)
    
    # Copy data from input tensor 1 (channels1)
    in_offset1 = ((b * channels1 + 0) * height + h) * width + w
    tile1 = tl.load(ptr1 + in_offset1)
    tl.store(out_ptr + out_offset + channels0, tile1)
    
    # Copy data from input tensor 2 (channels2)
    in_offset2 = ((b * channels2 + 0) * height + h) * width + w
    tile2 = tl.load(ptr2 + in_offset2)
    tl.store(out_ptr + out_offset + channels0 + channels1, tile2)
    
    # Copy data from input tensor 3 (channels3)
    in_offset3 = ((b * channels3 + 0) * height + h) * width + w
    tile3 = tl.load(ptr3 + in_offset3)
    tl.store(out_ptr + out_offset + channels0 + channels1 + channels2, tile3)
    
    # Copy data from input tensor 4 (channels4)
    in_offset4 = ((b * channels4 + 0) * height + h) * width + w
    tile4 = tl.load(ptr4 + in_offset4)
    tl.store(out_ptr + out_offset + channels0 + channels1 + channels2 + channels3, tile4)

@torch.fx.wrap
def optimized_concat_features(in_5, in_7, in_8, in_6, tmp_7):
    # Basic concatenation using resize_and_copy operations
    # Input shapes: [1, 2048, 64, 64], [1, 512, 64, 64], [1, 512, 64, 64], [1, 512, 64, 64], [1, 512, 64, 64]
    # Output should be [1, 2048+512*4, 64, 64] = [1, 3072, 64, 64]
    
    batch, c1, h, w = in_5.shape
    c2, c3, c4, c5 = in_7.shape[1], in_8.shape[1], in_6.shape[1], tmp_7.shape[1]
    total_channels = c1 + c2 + c3 + c4 + c5
    
    # Create output tensor
    output = torch.empty((batch, total_channels, h, w), dtype=in_5.dtype, device=in_5.device)
    
    # Copy each tensor to the output using safe operations
    # Tensor 1: channels 0-2047
    output[:, :c1, :, :] = in_5
    
    # Tensor 2: channels 2048-2559  
    output[:, c1:c1+c2, :, :] = in_7
    
    # Tensor 3: channels 2560-3071
    output[:, c1+c2:c1+c2+c3, :, :] = in_8
    
    # Tensor 4: channels 3072-3583 (Wait, let me recalculate...)
    output[:, c1+c2+c3:c1+c2+c3+c4, :, :] = in_6
    
    # Tensor 5: channels 3584-4095
    output[:, c1+c2+c3+c4:, :, :] = tmp_7
    
    return output

def replacement_func():
    return optimized_concat_features