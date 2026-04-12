import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    # Match the exact computation pattern from the model
    tmp_0 = torch.nn.functional.unfold(in_1, kernel_size=(384, 384), stride=(192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    tmp_3 = torch.nn.functional.unfold(in_2, kernel_size=(384, 384), stride=(288, 288))
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim=0)
    tmp_7 = tmp_6.to(dtype=torch.float16)
    return tmp_7

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Triton kernel for fused patch extraction and concatenation
@triton.jit
def fused_patch_extraction_kernel(
    in0_ptr, in0_batch, in0_channels, in0_h, in0_w,
    in1_ptr, in1_batch, in1_channels, in1_h, in1_w,
    in2_ptr, in2_batch, in2_channels, in2_h, in2_w,
    out_ptr,   # final output
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate total patches for each input
    in1_patches_h = (in1_h - 384) // 192 + 1
    in1_patches_w = (in1_w - 384) // 192 + 1
    in1_total_patches = in1_patches_h * in1_patches_w
    
    in2_patches_h = (in2_h - 384) // 288 + 1
    in2_patches_w = (in2_w - 384) // 288 + 1
    in2_total_patches = in2_patches_h * in2_patches_w
    
    # Total elements in output (patches + original)
    total_elements = (in1_total_patches + in2_total_patches + 1) * in0_channels * 384 * 384
    
    if pid >= total_elements:
        return
    
    # Determine which tensor element we're processing
    flat_idx = pid
    
    # Handle original input (in_0) - just copy
    if flat_idx < in0_channels * 384 * 384:
        out_idx = flat_idx
        c = out_idx // (384 * 384)
        h = (out_idx % (384 * 384)) // 384
        w = out_idx % 384
        val = tl.load(in0_ptr + c * in0_h * in0_w + h * in0_w + w)
        tl.store(out_ptr + out_idx, val.to(tl.float16))
        return
    
    flat_idx -= in0_channels * 384 * 384
    
    # Handle in_1 patches
    if flat_idx < in1_total_patches * in1_channels * 384 * 384:
        patch_idx = flat_idx // (in1_channels * 384 * 384)
        elem_idx = flat_idx % (in1_channels * 384 * 384)
        
        patch_h = patch_idx // in1_patches_w
        patch_w = patch_idx % in1_patches_w
        c = elem_idx // (384 * 384)
        ph = (elem_idx % (384 * 384)) // 384
        pw = elem_idx % 384
        
        src_h = patch_h * 192 + ph
        src_w = patch_w * 192 + pw
        if src_h < in1_h and src_w < in1_w:
            val = tl.load(in1_ptr + c * in1_h * in1_w + src_h * in1_w + src_w)
            out_flat_idx = in0_channels * 384 * 384 + patch_idx * in1_channels * 384 * 384 + elem_idx
            tl.store(out_ptr + out_flat_idx, val.to(tl.float16))
        return
    
    flat_idx -= in1_total_patches * in1_channels * 384 * 384
    
    # Handle in_2 patches
    patch_idx = flat_idx // (in2_channels * 384 * 384)
    elem_idx = flat_idx % (in2_channels * 384 * 384)
    
    patch_h = patch_idx // in2_patches_w
    patch_w = patch_idx % in2_patches_w
    c = elem_idx // (384 * 384)
    ph = (elem_idx % (384 * 384)) // 384
    pw = elem_idx % 384
    
    src_h = patch_h * 288 + ph
    src_w = patch_w * 288 + pw
    if src_h < in2_h and src_w < in2_w:
        val = tl.load(in2_ptr + c * in2_h * in2_w + src_h * in2_w + src_w)
        out_flat_idx = (in0_channels * 384 * 384 + 
                      in1_total_patches * in1_channels * 384 * 384 + 
                      patch_idx * in2_channels * 384 * 384 + 
                      elem_idx)
        tl.store(out_ptr + out_flat_idx, val.to(tl.float16))

# Kernel wrapper with automatic type conversion
@torch.fx.wrap
def fused_patch_extraction(in_0, in_1, in_2):
    # Calculate total patches for each input
    in1_patches_h = (in_1.shape[2] - 384) // 192 + 1
    in1_patches_w = (in_1.shape[3] - 384) // 192 + 1
    in1_total_patches = in1_patches_h * in1_patches_w
    
    in2_patches_h = (in_2.shape[2] - 384) // 288 + 1
    in2_patches_w = (in_2.shape[3] - 384) // 288 + 1
    in2_total_patches = in2_patches_h * in2_patches_w
    
    # Total elements for output
    total_elements = (in1_total_patches + in2_total_patches + 1) * in_0.shape[1] * 384 * 384
    
    # Output tensor in float16 as required by the original computation
    output = torch.empty(total_elements, dtype=torch.float16, device=in_0.device)
    
    # Calculate grid dimensions
    BLOCK_SIZE = 1024  # Optimized block size for GPU
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_patch_extraction_kernel[(num_programs,)](
        in0_ptr=in_0,
        in0_batch=in_0.shape[0],
        in0_channels=in_0.shape[1],
        in0_h=in_0.shape[2], 
        in0_w=in_0.shape[3],
        in1_ptr=in_1,
        in1_batch=in_1.shape[0],
        in1_channels=in_1.shape[1],
        in1_h=in_1.shape[2], 
        in1_w=in_1.shape[3],
        in2_ptr=in_2,
        in2_batch=in_2.shape[0],
        in2_channels=in_2.shape[1],
        in2_h=in_2.shape[2],
        in2_w=in_2.shape[3],
        out_ptr=output,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output to match expected format: [total_patches + 1, 3, 384, 384]
    total_patches = in1_total_patches + in2_total_patches + 1
    return output.reshape(total_patches, in_0.shape[1], 384, 384)

# Replacement function (must return function reference)
def replacement_func():
    return fused_patch_extraction