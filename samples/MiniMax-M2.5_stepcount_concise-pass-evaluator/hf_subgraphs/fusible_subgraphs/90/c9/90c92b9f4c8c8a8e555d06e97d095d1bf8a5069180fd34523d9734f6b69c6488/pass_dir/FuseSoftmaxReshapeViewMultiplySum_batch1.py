import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern: Fused softmax-reshape-view-multiply-sum operation for batch=1
    This matches the computation:
    1. softmax(in_1, dim=1)
    2. reshape to [1, -1]
    3. view to [1, -1, 1, 1]
    4. view to [1, 2, -1, 1, 1]
    5. multiply with in_0
    6. sum over dim=1
    7. contiguous
    
    Using exact operations as they appear in the FX graph.
    """
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(1, -1)
    tmp_2 = tmp_1.view(1, -1, 1, 1)
    tmp_3 = tmp_2.view(1, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_softmax_weighted_sum_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    batch_size, num_channels, feature_dim, height, width,
    stride_in0_b, stride_in0_c, stride_in0_h, stride_in0_w,
    stride_in1_b, stride_in1_c, stride_in1_h, stride_in1_w,
    stride_out_b, stride_out_h, stride_out_w,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_HW: tl.constexpr
):
    """
    Fused kernel that computes:
    1. Softmax over dim=1 of in_1
    2. Multiply with in_0
    3. Sum over channel dimension
    
    in_0: [batch, 2, 128, H, W]
    in_1: [batch, 2, 1, 128]
    out: [batch, H, W]
    """
    # Get batch and spatial position
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    # Calculate height * width
    hw_range = height * width
    
    # Calculate h and w from pid_hw
    h = pid_hw // width
    w = pid_hw % width
    
    # Load in_1 and compute softmax over channel dimension
    # in_1 has shape [batch, 2, 1, 128]
    # We need to compute softmax over dim=1 (the 2 channels)
    
    # Load values for both channels
    in1_base = in_1_ptr + pid_b * stride_in1_b
    
    # Load the 128 values for channel 0
    vals_0 = tl.load(in1_base + 0 * stride_in1_c + 0 * stride_in1_h + tl.arange(0, 128) * stride_in1_w).to(tl.float32)
    
    # Load the 128 values for channel 1  
    vals_1 = tl.load(in1_base + 1 * stride_in1_c + 0 * stride_in1_h + tl.arange(0, 128) * stride_in1_w).to(tl.float32)
    
    # Compute softmax: exp(vals) / sum(exp(vals)) over channels
    # For each feature dim position, compute softmax over the 2 channels
    softmax_0 = tl.zeros((128,), tl.float32)
    softmax_1 = tl.zeros((128,), tl.float32)
    
    # Compute exp for both channels
    exp_0 = tl.exp(vals_0)
    exp_1 = tl.exp(vals_1)
    
    # Sum exp values across channels
    sum_exp = exp_0 + exp_1 + 1e-8  # small epsilon for numerical stability
    
    # Compute softmax
    softmax_0 = exp_0 / sum_exp
    softmax_1 = exp_1 / sum_exp
    
    # Now compute weighted sum: sum over channels of (softmax * in_0)
    # in_0: [batch, 2, 128, H, W]
    # For each spatial position (h, w), compute sum over channels and feature_dim
    
    result = tl.zeros((1,), tl.float32)
    
    # Channel 0 contribution
    in0_base_0 = in_0_ptr + pid_b * stride_in0_b + 0 * stride_in0_c + h * stride_in0_h + w * stride_in0_w
    for f in range(128):
        in0_val = tl.load(in0_base_0 + f * stride_in0_c).to(tl.float32)
        result += softmax_0[f] * in0_val
    
    # Channel 1 contribution
    in0_base_1 = in_0_ptr + pid_b * stride_in0_b + 1 * stride_in0_c + h * stride_in0_h + w * stride_out_w
    for f in range(128):
        in0_val = tl.load(in0_base_1 + f * stride_in0_c).to(tl.float32)
        result += softmax_1[f] * in0_val
    
    # Store result
    out_ptr_offset = pid_b * stride_out_b + h * stride_out_h + w * stride_out_w
    tl.store(out_ptr + out_ptr_offset, result)


@torch.fx.wrap
def fused_softmax_weighted_sum_wrapper(in_0, in_1):
    """
    Wrapper function for the fused kernel.
    
    in_0: [batch, 2, 128, H, W] - value tensor
    in_1: [batch, 2, 1, 128] - attention/scores tensor
    
    Returns: [batch, H, W] - weighted sum output
    """
    batch_size = in_0.shape[0]
    num_channels = in_0.shape[1]  # 2
    feature_dim = in_0.shape[2]   # 128
    height = in_0.shape[3]
    width = in_0.shape[4]
    
    # Output shape
    out = torch.empty((batch_size, height, width), dtype=torch.float32, device=in_0.device)
    
    # Define block sizes
    BLOCK_SIZE_B = 1  # One batch element per program
    BLOCK_SIZE_HW = 256  # Process multiple spatial positions
    
    # Calculate grid
    grid = (batch_size, (height * width + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW)
    
    fused_softmax_weighted_sum_kernel[grid](
        in_0, in_1, out,
        batch_size, num_channels, feature_dim, height, width,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SIZE_B, BLOCK_SIZE_HW
    )
    
    return out


def replacement_func():
    return fused_softmax_weighted_sum_wrapper