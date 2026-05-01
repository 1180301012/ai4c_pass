import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    return tmp_11


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def normalize_kernel(
    in_0_ptr, in_1_ptr,
    out_ptr,
    B, H, W,
    in0_stride0, in0_stride1, in0_stride2, in0_stride3,
    in1_stride0, in1_stride1, in1_stride2, in1_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Get the program IDs
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    
    start_h = h * BLOCK_H
    start_w = w * BLOCK_W
    
    # Thread ID in block
    thread_h = tl.thread_id(0)
    thread_w = tl.thread_id(1)
    
    h_idx = start_h + thread_h
    w_idx = start_w + thread_w
    if h_idx >= H or w_idx >= W:
        return

    # Compute input offsets
    in0_1_offset = b * in0_stride0 + 1 * in0_stride1 + h_idx * in0_stride2 + w_idx * in0_stride3
    in0_2_offset = b * in0_stride0 + 2 * in0_stride1 + h_idx * in0_stride2 + w_idx * in0_stride3
    in1_0_offset = b * in1_stride0 + 0 * in1_stride1 + h_idx * in1_stride2 + w_idx * in1_stride3

    # Load input values
    x1 = tl.load(in_0_ptr + in0_1_offset)
    x2 = tl.load(in_0_ptr + in0_2_offset)
    x0 = tl.load(in_1_ptr + in1_0_offset)

    # Apply normalization
    y0 = x0 * 0.458 - 0.03
    y1 = x1 * 0.448 - 0.088
    y2 = x2 * 0.45 - 0.188

    # Calculate output offsets
    out0_offset = b * out_stride0 + 0 * out_stride1 + h_idx * out_stride2 + w_idx * out_stride3
    out1_offset = b * out_stride0 + 1 * out_stride1 + h_idx * out_stride2 + w_idx * out_stride3
    out2_offset = b * out_stride0 + 2 * out_stride1 + h_idx * out_stride2 + w_idx * out_stride3

    # Store results
    tl.store(out_ptr + out0_offset, y0)
    tl.store(out_ptr + out1_offset, y1)
    tl.store(out_ptr + out2_offset, y2)


@torch.fx.wrap
def normalized_tensor(in_0, in_1):
    B, C, H, W = in_0.shape
    assert C >= 3, "in_0 must have at least 3 channels"
    
    out = torch.empty((B, 3, H, W), dtype=in_0.dtype, device=in_0.device)
    
    # Common block sizes for optimal GPU utilization
    BLOCK_H = 32
    BLOCK_W = 32
    
    # Calculate grid dimensions
    grid = (B, (H + BLOCK_H - 1) // BLOCK_H, (W + BLOCK_W - 1) // BLOCK_W)
    
    # Get strides
    in0_stride0, in0_stride1, in0_stride2, in0_stride3 = in_0.stride()
    in1_stride0, in1_stride1, in1_stride2, in1_stride3 = in_1.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()

    # Launch kernel
    normalize_kernel[grid](
        in_0, in_1,
        out,
        B, H, W,
        in0_stride0, in0_stride1, in0_stride2, in0_stride3,
        in1_stride0, in1_stride1, in1_stride2, in1_stride3,
        out_stride0, out_stride1, out_stride2, out_stride3,
        BLOCK_H, BLOCK_W
    )
    
    return out


def replacement_func():
    return normalized_tensor