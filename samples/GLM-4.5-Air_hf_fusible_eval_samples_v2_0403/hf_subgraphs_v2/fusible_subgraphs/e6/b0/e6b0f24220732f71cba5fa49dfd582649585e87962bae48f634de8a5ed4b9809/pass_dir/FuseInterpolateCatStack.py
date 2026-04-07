import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat((in_2, in_3), 1)
    tmp_1 = torch.nn.functional.interpolate(in_0, size=(40, 40), mode='nearest')
    tmp_2 = torch.nn.functional.interpolate(in_1, size=(40, 40), mode='nearest')
    tmp_3 = torch.stack([tmp_1, tmp_2, tmp_0])
    return (tmp_3,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    out_ptr,
    batch_size,
    channels_0,
    channels_1,
    channels_23,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    1. Nearest interpolate in_0 (already 40x40, identity)
    2. Nearest interpolate in_1 (20x20 -> 40x40, 2x upsample)
    3. Cat (in_2, in_3) along channel dim
    4. Stack results [interpolate(in_0), interpolate(in_1), cat(in_2, in_3)]
    
    Output shape: [3, batch, channels, 40, 40]
    where channels_0=512, channels_1=512, channels_23=512
    """
    batch = tl.program_id(0)
    slice_dim = tl.program_id(1)  # 0, 1, or 2
    channel_idx = tl.program_id(2)
    
    block_start = tl.program_id(3) * BLOCK_SIZE
    h = tl.program_id(4)
    w = tl.program_id(5)
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < width
    
    bh, bw = batch * height * width, h * width + w
    
    # Calculate position for output
    out_channel_offset = slice_dim * channels_0 * batch_size * height * width
    out_bh_offset = channel_idx * batch_size * height * width
    out_h_offset = h * width
    pos = out_channel_offset + out_bh_offset + out_h_offset + offsets
    
    out_ptr_pos = out_ptr + tl.where(mask, pos, 0)
    
    if slice_dim == 0:
        # in_0 is already 40x40 (identity interpolate)
        in_h = h
        in_w = tl.where(mask, tl.cast(w, tl.int32), 0)
        in_pos = (in_0_ptr + 
                 batch * channels_0 * height * width +
                 channel_idx * height * width +
                 in_h * width + in_w)
        val = tl.load(in_pos, mask=mask, other=0.0)
    elif slice_dim == 1:
        # in_1 is 20x20 -> 40x40 (2x nearest upsample)
        in_h = h // 2
        in_w = tl.where(mask, tl.cast(w // 2, tl.int32), 0)
        in_pos = (in_1_ptr +
                 batch * channels_1 * (height // 2) * (width // 2) +
                 channel_idx * (height // 2) * (width // 2) +
                 in_h * (width // 2) + in_w)
        val = tl.load(in_pos, mask=mask, other=0.0)
    else:
        # in_2 and in_3 are concatenated along channel dim
        # each has channels_23 = 256 channels
        if channel_idx < channels_23:
            # from in_2
            in_pos = (in_2_ptr +
                     batch * channels_23 * height * width +
                     channel_idx * height * width +
                     h * width + w)
            val = tl.load(in_pos, mask=mask, other=0.0)
        else:
            # from in_3
            in_idx = channel_idx - channels_23
            in_pos = (in_3_ptr +
                     batch * channels_23 * height * width +
                     in_idx * height * width +
                     h * width + w)
            val = tl.load(in_pos, mask=mask, other=0.0)
    
    tl.store(out_ptr_pos, val, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2, in_3):
    batch_size = in_0.shape[0]
    channels_0 = in_0.shape[1]
    channels_1 = in_1.shape[1]
    channels_23 = in_2.shape[1]  # 256
    height, width = 40, 40
    
    # Output: [3, batch, 1024, 40, 40]
    # channels: 512 + 512 + (256+256) = 1536 total but stacked as 3 slices
    out = torch.empty((3, batch_size, channels_0 + channels_1, height, width), 
                      dtype=in_0.dtype, device=in_0.device)
    
    n_elems = height * width
    BLOCK_SIZE = triton.next_power_of_2(n_elems)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    grid = (
        batch_size,
        3,  # 3 slices
        channels_0 + channels_1,
        1,  # We do this in a single pass through channels
        1,  # height
        1,  # width  
    )
    
    # We need to launch for each height, width position
    # Reconsider the grid structure
    num_channel_tiles = 1
    
    fused_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        batch_size=batch_size,
        channels_0=channels_0,
        channels_1=channels_1,
        channels_23=channels_23,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out,)


def replacement_func():
    return fused_kernel_wrapper