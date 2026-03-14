import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Match two consecutive bilinear interpolate operations
    """
    tmp_0 = torch.nn.functional.interpolate(in_0, (32, 32), None, 'bilinear', False)
    tmp_1 = torch.nn.functional.interpolate(in_1, (32, 32), None, 'bilinear', False)
    return (tmp_0, tmp_1)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_bilinear_interpolate_kernel(
    in0_ptr, in1_ptr,
    out0_ptr, out1_ptr,
    batch, channels, in_h, in_w,
    out_h, out_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused bilinear interpolation kernel for two tensors
    """
    pid = tl.program_id(0)
    
    # Calculate which position we're working on
    total_elements = batch * channels * out_h * out_w
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements
    
    # Decompose linear index to 4D coordinates (b, c, h, w)
    w_out = offset % out_w
    h_out = (offset // out_w) % out_h
    c = (offset // (out_w * out_h)) % channels
    b = offset // (out_w * out_h * channels)
    
    # Calculate input coordinates for bilinear interpolation
    # scale = in_size / out_size
    scale_h = tl.cast(in_h, tl.float32) / tl.cast(out_h, tl.float32)
    scale_w = tl.cast(in_w, tl.float32) / tl.cast(out_w, tl.float32)
    
    # Get source coordinates (align_corners=False)
    src_y = (tl.cast(h_out, tl.float32) + 0.5) * scale_h - 0.5
    src_x = (tl.cast(w_out, tl.float32) + 0.5) * scale_w - 0.5
    
    # Clamp to valid range
    src_y = tl.maximum(0.0, src_y)
    src_x = tl.maximum(0.0, src_x)
    
    # Get integer and fractional parts
    y0 = tl.cast(src_y, tl.int32)
    x0 = tl.cast(src_x, tl.int32)
    y1 = tl.minimum(y0 + 1, in_h - 1)
    x1 = tl.minimum(x0 + 1, in_w - 1)
    y0 = tl.minimum(y0, in_h - 1)
    x0 = tl.minimum(x0, in_w - 1)
    
    wy1 = src_y - tl.cast(y0, tl.float32)
    wx1 = src_x - tl.cast(x0, tl.float32)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1
    
    # Calculate input indices
    base_idx = b * channels * in_h * in_w + c * in_h * in_w
    idx_00 = base_idx + y0 * in_w + x0
    idx_01 = base_idx + y0 * in_w + x1
    idx_10 = base_idx + y1 * in_w + x0
    idx_11 = base_idx + y1 * in_w + x1
    
    # Load and interpolate for first tensor
    v00_0 = tl.load(in0_ptr + idx_00, mask=mask, other=0.0)
    v01_0 = tl.load(in0_ptr + idx_01, mask=mask, other=0.0)
    v10_0 = tl.load(in0_ptr + idx_10, mask=mask, other=0.0)
    v11_0 = tl.load(in0_ptr + idx_11, mask=mask, other=0.0)
    
    out_val_0 = (wy0 * wx0 * v00_0 + wy0 * wx1 * v01_0 + 
                 wy1 * wx0 * v10_0 + wy1 * wx1 * v11_0)
    
    # Load and interpolate for second tensor
    v00_1 = tl.load(in1_ptr + idx_00, mask=mask, other=0.0)
    v01_1 = tl.load(in1_ptr + idx_01, mask=mask, other=0.0)
    v10_1 = tl.load(in1_ptr + idx_10, mask=mask, other=0.0)
    v11_1 = tl.load(in1_ptr + idx_11, mask=mask, other=0.0)
    
    out_val_1 = (wy0 * wx0 * v00_1 + wy0 * wx1 * v01_1 + 
                 wy1 * wx0 * v10_1 + wy1 * wx1 * v11_1)
    
    # Store results
    tl.store(out0_ptr + offset, out_val_0, mask=mask)
    tl.store(out1_ptr + offset, out_val_1, mask=mask)

@torch.fx.wrap
def fused_bilinear_interpolate(in_0, in_1):
    """
    Fused bilinear interpolation for two tensors
    """
    batch, channels, in_h, in_w = in_0.shape
    out_h, out_w = 32, 32
    
    # Create output tensors
    out_0 = torch.empty((batch, channels, out_h, out_w), 
                        dtype=in_0.dtype, device=in_0.device)
    out_1 = torch.empty((batch, channels, out_h, out_w), 
                        dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel
    total_elements = batch * channels * out_h * out_w
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_bilinear_interpolate_kernel[grid](
        in_0, in_1,
        out_0, out_1,
        batch, channels, in_h, in_w,
        out_h, out_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_0, out_1

def replacement_func():
    return fused_bilinear_interpolate