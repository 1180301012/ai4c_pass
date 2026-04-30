import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_relu_maxpool_cat_kernel(
    input_ptr, output_ptr,
    B, C_IN, H, W, C_OUT,
    stride_ib, stride_ic, stride_ih, stride_iw,
    stride_ob, stride_oc, stride_oh, stride_ow,
    TH: tl.constexpr, TW: tl.constexpr,
):
    """
    Fused kernel for: ReLU + 3x MaxPool2d(5,1,2,1) + Cat along dim 1
    
    Optimization: Since all 3 max_pool2d operations have identical parameters
    and input, we compute max_pool only once and write it 3 times to the output.
    ReLU is also fused, eliminating intermediate tensor allocations.
    
    Grid: (B * C_IN, num_h_blocks, num_w_blocks)
    Each program handles a TH x TW tile of spatial positions for a specific (b, c_in).
    """
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Decode program ID to (b, c_in)
    b = pid_bc // C_IN
    c_in = pid_bc % C_IN
    
    # Tile start positions
    h_start = pid_h * TH
    w_start = pid_w * TW
    
    # Input region needed: (TH+4) x (TW+4) covering the 5x5 windows
    # for all TH x TW output positions in the tile
    ih_range = h_start - 2 + tl.arange(0, TH + 4)
    iw_range = w_start - 2 + tl.arange(0, TW + 4)
    
    # Clamp coordinates to valid range [0, H-1] and [0, W-1]
    # This is equivalent to 0-padding for max_pool on relu'd input because:
    # - After ReLU, all values are >= 0
    # - Padding fills with 0, which doesn't affect max (since max(relu_vals) >= 0)
    # - Edge values (from clamping) are >= 0 and are a subset of valid values
    ih_clamped = tl.where(ih_range < 0, 0, ih_range)
    ih_clamped = tl.where(ih_clamped >= H, H - 1, ih_clamped)
    iw_clamped = tl.where(iw_range < 0, 0, iw_range)
    iw_clamped = tl.where(iw_clamped >= W, W - 1, iw_clamped)
    
    # 2D clamped coordinates for loading
    ih_2d = ih_clamped[:, None]  # (TH+4, 1)
    iw_2d = iw_clamped[None, :]  # (1, TW+4)
    
    # Load input region and cast to float32 for computation
    in_offsets = b * stride_ib + c_in * stride_ic + ih_2d * stride_ih + iw_2d * stride_iw
    input_block = tl.load(input_ptr + in_offsets).to(tl.float32)
    
    # Apply ReLU
    relu_block = tl.maximum(input_block, 0.0)
    
    # Compute max_pool2d for each output position in the tile
    # The 5x5 window for position (dh_out, dw_out) in the tile
    # starts at (dh_out, dw_out) in the loaded block
    pool_result = tl.full([TH, TW], 0.0, dtype=tl.float32)
    
    for dh in range(5):
        for dw in range(5):
            window = relu_block[dh:dh+TH, dw:dw+TW]
            pool_result = tl.maximum(pool_result, window)
    
    # ReLU center values (for the non-pooled output channel)
    # Center position in the block is at offset (2, 2)
    relu_center = relu_block[2:2+TH, 2:2+TW]
    
    # Output positions and validity mask
    h_out_range = h_start + tl.arange(0, TH)
    w_out_range = w_start + tl.arange(0, TW)
    h_out_valid = (h_out_range >= 0) & (h_out_range < H)
    w_out_valid = (w_out_range >= 0) & (w_out_range < W)
    out_mask = h_out_valid[:, None] & w_out_valid[None, :]
    
    h_out_2d = h_out_range[:, None]
    w_out_2d = w_out_range[None, :]
    
    # Base output offset for channel c_in
    out_base = b * stride_ob + c_in * stride_oc + h_out_2d * stride_oh + w_out_2d * stride_ow
    
    # Store relu result at channel c_in (the non-pooled branch)
    tl.store(output_ptr + out_base, relu_center, mask=out_mask)
    
    # Store pool result at channels c_in+C_IN, c_in+2*C_IN, c_in+3*C_IN
    # Since all 3 max_pool operations are identical, we write the same result 3 times
    pool_stride = C_IN * stride_oc
    tl.store(output_ptr + out_base + pool_stride, pool_result, mask=out_mask)
    tl.store(output_ptr + out_base + 2 * pool_stride, pool_result, mask=out_mask)
    tl.store(output_ptr + out_base + 3 * pool_stride, pool_result, mask=out_mask)


@torch.fx.wrap
def fused_relu_maxpool_cat(input_tensor):
    """
    Fused implementation of ReLU + 3x MaxPool2d(5,1,2,1) + Cat(in_0, pool, pool, pool, dim=1)
    
    Output shape: (B, 4*C_IN, H, W) where H and W are preserved (stride=1, padding=2, kernel=5)
    """
    B, C_IN, H, W = input_tensor.shape
    C_OUT = C_IN * 4
    
    output_tensor = torch.empty((B, C_OUT, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Tile sizes - each program processes a TH x TW spatial tile
    TH = 8
    TW = 8
    
    stride_ib, stride_ic, stride_ih, stride_iw = input_tensor.stride()
    stride_ob, stride_oc, stride_oh, stride_ow = output_tensor.stride()
    
    # Grid configuration
    num_bc = B * C_IN
    num_h_blocks = (H + TH - 1) // TH
    num_w_blocks = (W + TW - 1) // TW
    
    grid = (num_bc, num_h_blocks, num_w_blocks)
    
    fused_relu_maxpool_cat_kernel[grid](
        input_tensor, output_tensor,
        B, C_IN, H, W, C_OUT,
        stride_ib, stride_ic, stride_ih, stride_iw,
        stride_ob, stride_oc, stride_oh, stride_ow,
        TH=TH, TW=TW,
    )
    
    return output_tensor


def replacement_func():
    return fused_relu_maxpool_cat