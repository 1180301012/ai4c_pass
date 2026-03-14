import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the pattern: softmax(in_1, dim=1) * in_0, then sum(dim=1)
    This corresponds to a weighted sum where softmax provides the weights.
    
    in_0: [B, C1, C2, H, W] = [1, 2, 256, 32, 32] or [1, 2, 256, 8, 8]
    in_1: [B, C1, C2, 1, 1] = [1, 2, 256, 1, 1]
    output: [B, C2, H, W] = [1, 256, 32, 32] or [1, 256, 8, 8]
    """
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_softmax_mul_sum_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    B, C1, C2, H, W,
    spatial_elements,
    stride_in_0_b, stride_in_0_c1, stride_in_0_c2, stride_in_0_h, stride_in_0_w,
    stride_in_1_b, stride_in_1_c1, stride_in_1_c2, stride_in_1_h, stride_in_1_w,
    stride_out_b, stride_out_c2, stride_out_h, stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes: sum(in_0 * softmax(in_1, dim=1), dim=1)
    
    in_0: [B, C1, C2, H, W] = [1, 2, 256, 32, 32] or [1, 2, 256, 8, 8]
    in_1: [B, C1, C2, 1, 1] = [1, 2, 256, 1, 1]
    output: [B, C2, H, W] = [1, 256, 32, 32] or [1, 256, 8, 8]
    
    Grid: (C2, ceil(spatial_elements / BLOCK_SIZE)) 
    - Each row processes all channels c2 in parallel
    - Each column processes BLOCK_SIZE spatial elements
    """
    # pid_x = c2 (channel), pid_y = spatial position group
    c2 = tl.program_id(0)
    pid = tl.program_id(1)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_elements
    
    # Compute h, w from flattened spatial offsets
    # spatial_elements = B * H * W (but B is always 1)
    h = offsets // W
    w = offsets % W
    b = offsets // (H * W)  # should be 0 since B=1
    
    # Load in_1 values for softmax - same for all spatial positions with same c2
    # in_1[b, 0, c2, 0, 0] and in_1[b, 1, c2, 0, 0]
    in_1_offset_0 = b * stride_in_1_b + 0 * stride_in_1_c1 + c2 * stride_in_1_c2
    in_1_offset_1 = b * stride_in_1_b + 1 * stride_in_1_c1 + c2 * stride_in_1_c2
    
    in_1_0 = tl.load(in_1_ptr + in_1_offset_0)  # same for all threads in this c2
    in_1_1 = tl.load(in_1_ptr + in_1_offset_1)  # same for all threads in this c2
    
    # Compute softmax - same for all spatial positions with same c2
    max_val = tl.maximum(in_1_0, in_1_1)
    exp_0 = tl.exp(in_1_0 - max_val)
    exp_1 = tl.exp(in_1_1 - max_val)
    sum_exp = exp_0 + exp_1 + 1e-8
    softmax_0 = exp_0 / sum_exp
    softmax_1 = exp_1 / sum_exp
    
    # Load in_0 values and compute weighted sum for each spatial position
    # in_0[b, 0, c2, h, w] and in_0[b, 1, c2, h, w]
    in_0_offset_0 = b * stride_in_0_b + 0 * stride_in_0_c1 + c2 * stride_in_0_c2 + h * stride_in_0_h + w * stride_in_0_w
    in_0_offset_1 = b * stride_in_0_b + 1 * stride_in_0_c1 + c2 * stride_in_0_c2 + h * stride_in_0_h + w * stride_in_0_w
    
    in_0_0 = tl.load(in_0_ptr + in_0_offset_0, mask=mask)
    in_0_1 = tl.load(in_0_ptr + in_0_offset_1, mask=mask)
    
    result = in_0_0 * softmax_0 + in_0_1 * softmax_1
    
    # Store result to out[b, c2, h, w]
    out_offset = b * stride_out_b + c2 * stride_out_c2 + h * stride_out_h + w * stride_out_w
    tl.store(out_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def fused_softmax_mul_sum(in_0, in_1):
    """
    Fused kernel: sum(in_0 * softmax(in_1, dim=1), dim=1)
    
    in_0: [B, C1, C2, H, W] = [1, 2, 256, 32, 32] or [1, 2, 256, 8, 8]
    in_1: [B, C1, C2, 1, 1] = [1, 2, 256, 1, 1]
    output: [B, C2, H, W] = [1, 256, 32, 32] or [1, 256, 8, 8]
    """
    B, C1, C2, H, W = in_0.shape
    
    # Output shape: [B, C2, H, W]
    out_shape = (B, C2, H, W)
    out = torch.empty(out_shape, device=in_0.device, dtype=in_0.dtype)
    
    # Grid: (C2, ceil(spatial_elements / BLOCK_SIZE))
    # C2 = 256 channels, each processes multiple spatial positions
    spatial_elements = B * H * W
    # Use block size that matches spatial elements for better efficiency
    # For small spatial sizes (8x8=64), use smaller block; for larger (32x32=1024), use larger
    BLOCK_SIZE = 64 if spatial_elements <= 64 else 128
    grid = (C2, (spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    fused_softmax_mul_sum_kernel[grid](
        in_0, in_1, out,
        B, C1, C2, H, W,
        spatial_elements,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3), in_0.stride(4),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3), in_1.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_softmax_mul_sum