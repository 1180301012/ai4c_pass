import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact computation pattern from model.py
def pattern(in_0, in_1, in_2):
    # First unfold operation
    tmp_0 = torch.nn.functional.unfold(in_1, kernel_size = (384, 384), stride = (192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    
    # Second unfold operation
    tmp_3 = torch.nn.functional.unfold(in_2, kernel_size = (384, 384), stride = (288, 288))
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    
    # Concatenation and type conversion
    tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim = 0)
    tmp_7 = tmp_6.to(dtype = torch.float16)
    
    return tmp_7

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized Triton kernel that fuses the entire computation
@triton.jit
def fused_patch_extraction_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_ptr,
    in_0_batch, in_0_channels, in_0_h, in_0_w,
    in_1_batch, in_1_channels, in_1_h, in_1_w,
    in_2_batch, in_2_channels, in_2_h, in_2_w,
    out_batch, out_channels, out_h, out_w,
    PATCH_SIZE_H, PATCH_SIZE_W, STRIDE_H1, STRIDE_W1, STRIDE_H2, STRIDE_W2,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the number of patches for each unfold operation
    num_patches_1 = ((in_1_h - PATCH_SIZE_H) // STRIDE_H1 + 1) * ((in_1_w - STRIDE_W1) // STRIDE_W1 + 1)
    num_patches_2 = ((in_2_h - PATCH_SIZE_H) // STRIDE_H2 + 1) * ((in_2_w - STRIDE_W2) // STRIDE_W2 + 1)
    
    # Total batch size after concatenation
    total_patches = num_patches_1 + num_patches_2 + in_0_batch
    
    # Program ID mapping
    patch_idx = tl.program_id(0)
    
    if patch_idx >= total_patches:
        return
    
    # Determine which input this patch comes from
    if patch_idx < num_patches_2:
        # From in_2 (patches 0 to num_patches_2-1)
        # Calculate in_2 patch coordinates
        base_idx = patch_idx
        p_h = base_idx // ((in_2_w - PATCH_SIZE_W) // STRIDE_W2 + 1)
        p_w = base_idx % ((in_2_w - PATCH_SIZE_W) // STRIDE_W2 + 1)
        
        # Calculate starting position in input
        h_start = p_h * STRIDE_H2
        w_start = p_w * STRIDE_W2
        
        # Convert to absolute output index (in_2 patches come first)
        out_idx = patch_idx
        
    elif patch_idx < num_patches_2 + num_patches_1:
        # From in_1 (patches num_patches_2 to num_patches_2+num_patches_1-1)
        local_idx = patch_idx - num_patches_2
        # Calculate in_1 patch coordinates
        p_h = local_idx // ((in_1_w - PATCH_SIZE_W) // STRIDE_W1 + 1)
        p_w = local_idx % ((in_1_w - PATCH_SIZE_W) // STRIDE_W1 + 1)
        
        # Calculate starting position in input
        h_start = p_h * STRIDE_H1
        w_start = p_w * STRIDE_W1
        
        # Convert to absolute output index (in_1 patches)
        out_idx = num_patches_2 + local_idx
        
    else:
        # From in_0 (patches num_patches_2+num_patches_1 to end)
        local_idx = patch_idx - num_patches_2 - num_patches_1
        out_idx = patch_idx
        h_start = 0
        w_start = 0
    
    # Process each pixel in the patch
    pixel_idx = tl.program_id(1)
    channel_idx = tl.program_id(2)
    
    if (pixel_idx >= PATCH_SIZE_H * PATCH_SIZE_W or 
        channel_idx >= out_channels or 
        out_idx >= out_batch):
        return
    
    # Calculate patch pixel coordinates
    ph = pixel_idx // PATCH_SIZE_W
    pw = pixel_idx % PATCH_SIZE_W
    
    # Calculate output coordinates
    out_ph = h_start + ph
    out_pw = w_start + pw
    
    # Determine source coordinates and data type handling
    if out_idx < num_patches_2:
        # From in_2
        src_batch = 0
        src_channel = channel_idx
        src_ph = out_ph
        src_pw = out_pw
        src_ptr = in_2_ptr
        src_stride_h = in_2_w * in_2_channels
        src_stride_w = in_2_channels
    elif out_idx < num_patches_2 + num_patches_1:
        # From in_1
        src_batch = 0
        src_channel = channel_idx
        src_ph = out_ph
        src_pw = out_pw
        src_ptr = in_1_ptr
        src_stride_h = in_1_w * in_1_channels
        src_stride_w = in_1_channels
    else:
        # From in_0 (already in correct shape)
        src_batch = 0
        src_channel = channel_idx
        src_ph = ph  # Original coordinates for in_0
        src_pw = pw  # Original coordinates for in_0
        src_ptr = in_0_ptr
        src_stride_h = in_0_w * in_0_channels
        src_stride_w = in_0_channels
    
    if src_ptr == in_2_ptr:
        # in_2 tensor with shape [1, 3, 1536, 1536]
        src_stride_batch = in_2_channels * in_2_h * in_2_w
        src_stride_channel = in_2_h * in_2_w
        src_stride_pixel = in_2_channels
        offset = src_batch * src_stride_batch + src_ph * src_stride_channel + src_pw * src_stride_pixel + src_channel
    elif src_ptr == in_1_ptr:
        # in_1 tensor with shape [1, 3, 768, 768]
        src_stride_batch = in_1_channels * in_1_h * in_1_w
        src_stride_channel = in_1_h * in_1_w
        src_stride_pixel = in_1_channels
        offset = src_batch * src_stride_batch + src_ph * src_stride_channel + src_pw * src_stride_pixel + src_channel
    else:
        # in_0 tensor with shape [1, 3, 384, 384]
        src_stride_batch = in_0_channels * in_0_h * in_0_w
        src_stride_channel = in_0_h * in_0_w
        src_stride_pixel = in_0_channels
        offset = src_batch * src_stride_batch + src_ph * src_stride_channel + src_pw * src_stride_pixel + src_channel
    
    # Load data (handle bfloat16 to float16 conversion)
    src_val = tl.load(src_ptr + offset, mask=True)
    
    # Convert from bfloat16 to float16
    out_val = src_val if src_ptr == out_ptr else tl.bfloat16_to_float16(src_val)
    
    # Calculate output address
    out_offset = out_idx * out_channels * out_h * out_w + \
                 channel_idx * out_h * out_w + \
                 ph * out_w + pw
    
    # Store result
    tl.store(out_ptr + out_offset, out_val)

# Kernel wrapper with autotuning
@torch.fx.wrap
def fused_patch_extraction(in_0, in_1, in_2):
    # Input shapes
    in_0_shape = in_0.shape  # [1, 3, 384, 384]
    in_1_shape = in_1.shape  # [1, 3, 768, 768]
    in_2_shape = in_2.shape  # [1, 3, 1536, 1536]
    
    # Calculate output shape
    num_patches_1 = ((in_1_shape[2] - 384) // 192 + 1) * ((in_1_shape[3] - 384) // 192 + 1)  # 9
    num_patches_2 = ((in_2_shape[2] - 384) // 288 + 1) * ((in_2_shape[3] - 384) // 288 + 1)  # 25 not 37, let me recalculate:
    
    # Correct calculation for patches:
    # For in_2 [1, 3, 1536, 1536] with stride (288, 288):
    # height_patches = (1536 - 384) // 288 + 1 = (1152) // 288 + 1 = 4 + 1 = 5
    # width_patches = (1536 - 384) // 288 + 1 = (1152) // 288 + 1 = 4 + 1 = 5
    # total_patches = 5 * 5 = 25
    
    # Let me recalculate from the model.py - wait, let me check the original unfolding again:
    # For in_2 [1, 3, 1536, 1536], kernel=(384,384), stride=(288,288)
    # patches_h = (1536 - 384) // 288 + 1 = (1152) // 288 + 1 = 4 + 1 = 5
    # patches_w = (1536 - 384) // 288 + 1 = (1152) // 288 + 1 = 4 + 1 = 5
    # total patches = 5 * 5 = 25
    
    # But wait, let me trace the reshape operation again:
    # tmp_4 has shape [25, 1, 442368] after permute
    # reshape(-1, 3, 384, 384) gives [25, 3, 384, 384] since 25*1 = 25
    
    # Let me check the cat operation: [tmp_5, tmp_2, in_0] where:
    # tmp_5 = [25, 3, 384, 384] (from in_2)
    # tmp_2 = [9, 3, 384, 384] (from in_1)  
    # in_0 = [1, 3, 384, 384]
    
    # Resulting tensor shape becomes [35, 3, 384, 384]
    
    # Total output shape is [25+9+1, 3, 384, 384] = [35, 3, 384, 384]
    
    # Compute final patch counts
    num_patches_1 = ((in_1_shape[2] - 384) // 192 + 1) * ((in_1_shape[3] - 384) // 192 + 1)  # 3x3 = 9
    
    patches_h_in2 = ((in_2_shape[2] - 384) // 288 + 1)
    patches_w_in2 = ((in_2_shape[3] - 384) // 288 + 1)
    num_patches_2 = patches_h_in2 * patches_w_in2  # 5x5 = 25
    
    out_batch = num_patches_1 + num_patches_2 + in_0_shape[0]  # 9 + 25 + 1 = 35
    
    # Create output tensor
    out_shape = (out_batch, in_0_shape[1], 384, 384)
    out = torch.empty(out_shape, dtype=torch.float16, device=in_0.device)
    
    # Launch kernel with auto-tuned grid
    total_programs = out_batch * 384 * 384 * in_0_shape[1]
    BLOCK_SIZE = 128
    
    fused_patch_extraction_kernel[(out_batch, 384 * 384, in_0_shape[1])](
        in_0_ptr=in_0, in_1_ptr=in_1, in_2_ptr=in_2,
        out_ptr=out,
        in_0_batch=in_0_shape[0], in_0_channels=in_0_shape[1], in_0_h=in_0_shape[2], in_0_w=in_0_shape[3],
        in_1_batch=in_1_shape[0], in_1_channels=in_1_shape[1], in_1_h=in_1_shape[2], in_1_w=in_1_shape[3],
        in_2_batch=in_2_shape[0], in_2_channels=in_2_shape[1], in_2_h=in_2_shape[2], in_2_w=in_2_shape[3],
        out_batch=out_batch, out_channels=in_0_shape[1], out_h=384, out_w=384,
        PATCH_SIZE_H=384, PATCH_SIZE_W=384,
        STRIDE_H1=192, STRIDE_W1=192,
        STRIDE_H2=288, STRIDE_W2=288,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function (returns function reference, not a call)
def replacement_func():
    return fused_patch_extraction