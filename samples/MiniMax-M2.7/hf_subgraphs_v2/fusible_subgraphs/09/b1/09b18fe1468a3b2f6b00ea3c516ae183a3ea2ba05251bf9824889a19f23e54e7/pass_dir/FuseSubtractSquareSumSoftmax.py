import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_kernel(
    x_ptr, y_ptr, scale_ptr,
    output_ptr,
    stride_x0, stride_x1, stride_x2, stride_x3,
    stride_y0, stride_y1, stride_y2, stride_y3,
    stride_scale0, stride_scale1, stride_scale2,
    n_seg0, n_seg1, n_seg2, seg_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one segment of size seg_size
    # seg_idx = seg0 * n_seg1 * n_seg2 + seg1 * n_seg2 + seg2
    seg_idx = tl.program_id(0)
    
    # Compute segment coordinates
    seg0 = seg_idx // (n_seg1 * n_seg2)
    rem = seg_idx % (n_seg1 * n_seg2)
    seg1 = rem // n_seg2
    seg2 = rem % n_seg2
    
    # Compute offsets for this segment
    offsets_base = seg0 * stride_x0 + seg1 * stride_x1 + seg2 * stride_x2
    offsets = offsets_base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < offsets_base + seg_size
    
    # Load and compute (x - y)^2
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    diff = x - y
    diff_sq = diff * diff
    
    # Load scale [n_seg0=1, n_seg1=1, n_seg2=32] at seg2
    scale_offsets = seg2 * stride_scale2
    scale = tl.load(scale_ptr + scale_offsets)
    
    # Compute scaled value
    scaled = diff_sq.to(tl.float32) * scale.to(tl.float32)
    
    # Softmax computation
    # Find max for numerical stability
    max_val = tl.max(scaled, axis=0)
    max_val = tl.reshape(max_val, [1])
    
    # Compute exp(x - max)
    exp_val = tl.exp(scaled - max_val)
    
    # Compute sum of exp values
    sum_exp = tl.sum(exp_val, axis=0)
    sum_exp = tl.reshape(sum_exp, [1])
    
    # Compute softmax
    softmax_out = exp_val / sum_exp
    
    # Store result
    out_offsets = offsets_base
    tl.store(output_ptr + out_offsets, softmax_out.to(output_ptr.dtype.element_ty), mask=mask)


def fused_softmax(x, y, scale):
    # x: [1, 4096, 32, 512], y: [1, 1, 32, 512], scale: [1, 1, 32]
    # Output: [1, 4096, 32, 512]
    
    n_seg0, n_seg1, n_seg2, seg_size = x.shape
    
    # Use 1D grid with one program per segment
    num_segments = n_seg0 * n_seg1 * n_seg2
    
    # Allocate output
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 512  # seg_size
    
    fused_softmax_kernel[(num_segments,)](
        x_ptr=x, y_ptr=y, scale_ptr=scale,
        output_ptr=output,
        stride_x0=x.stride(0), stride_x1=x.stride(1), stride_x2=x.stride(2), stride_x3=x.stride(3),
        stride_y0=y.stride(0), stride_y1=y.stride(1), stride_y2=y.stride(2), stride_y3=y.stride(3),
        stride_scale0=scale.stride(0), stride_scale1=scale.stride(1), stride_scale2=scale.stride(2),
        n_seg0=n_seg0, n_seg1=n_seg1, n_seg2=n_seg2, seg_size=seg_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim = 3)
    tmp_4 = in_3 * tmp_3
    tmp_5 = torch.nn.functional.softmax(tmp_4, dim = 2)
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_9 = tmp_5.unsqueeze(3)
    tmp_10 = tmp_8 - tmp_6
    return tmp_10, tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return fused_softmax