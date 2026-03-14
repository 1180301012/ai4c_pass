import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern: Multiply + sum + contiguous
    This matches the final part of the computation.
    """
    tmp_4 = in_1 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_multiply_sum_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    batch_size, num_channels, feature_dim, height, width,
    stride_in0_b, stride_in0_c, stride_in0_f, stride_in0_h, stride_in0_w,
    stride_in1_b, stride_in1_c, stride_in1_f, 
    stride_out_b, stride_out_f, stride_out_h, stride_out_w,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel that computes: out = sum(in_1 * in_0, dim=1)
    
    in_0: [batch, 2, 128, H, W] - value features
    in_1: [batch, 2, 128, 1, 1] - softmax weights (shape after reshape/view)
    out: [batch, 128, H, W]
    """
    # Get position
    pid = tl.program_id(0)
    num_positions = batch_size * feature_dim * height * width
    
    if pid >= num_positions:
        return
    
    # Calculate batch, feature, h, w from linear index
    tmp = pid
    b = tmp // (feature_dim * height * width)
    tmp = tmp % (feature_dim * height * width)
    f = tmp // (height * width)
    tmp = tmp % (height * width)
    h = tmp // width
    w = tmp % width
    
    # Compute sum over channels: sum_c in_1[b, c, f, 0, 0] * in_0[b, c, f, h, w]
    result = 0.0
    
    for c in range(num_channels):
        # Load in_0[b, c, f, h, w]
        in0_idx = b * stride_in0_b + c * stride_in0_c + f * stride_in0_f + h * stride_in0_h + w * stride_in0_w
        in0_val = tl.load(in_0_ptr + in0_idx).to(tl.float32)
        
        # Load in_1[b, c, f, 0, 0] - in_1 has shape [batch, 2, 128, 1, 1], so h,w are always 0
        in1_idx = b * stride_in1_b + c * stride_in1_c + f * stride_in1_f
        in1_val = tl.load(in_1_ptr + in1_idx).to(tl.float32)
        
        result += in1_val * in0_val
    
    # Store result
    out_idx = b * stride_out_b + f * stride_out_f + h * stride_out_h + w * stride_out_w
    tl.store(out_ptr + out_idx, result)


@torch.fx.wrap
def fused_multiply_sum_wrapper(in_0, in_1):
    """
    Wrapper that calls the fused Triton kernel.
    
    in_0: [batch, 2, 128, H, W]
    in_1: [batch, 2, 128, 1, 1] (after softmax and reshape)
    
    Returns: [batch, 128, H, W]
    """
    batch_size = in_0.shape[0]
    num_channels = in_0.shape[1]  # 2
    feature_dim = in_0.shape[2]   # 128
    height = in_0.shape[3]
    width = in_0.shape[4]
    
    # Output shape: [batch, 128, H, W]
    out = torch.empty((batch_size, feature_dim, height, width), dtype=torch.float32, device=in_0.device)
    
    # Calculate grid
    num_positions = batch_size * feature_dim * height * width
    BLOCK_SIZE = 256
    grid = (num_positions,)
    
    fused_multiply_sum_kernel[grid](
        in_0, in_1, out,
        batch_size, num_channels, feature_dim, height, width,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3), in_0.stride(4),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE
    )
    
    return out


def replacement_func():
    return fused_multiply_sum_wrapper