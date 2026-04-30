import torch
import triton
import triton.language as tl


@triton.jit
def fused_softmax_weighted_sum_kernel(
    # Softmax input pointer (will be reshaped to [batch, 2, 128])
    softmax_ptr,
    # Weight input: [batch, 2, num_positions, H, W]
    weight_ptr,
    # Output: [batch, num_positions, H, W]
    out_ptr,
    # Dimensions
    batch: tl.constexpr,
    num_positions: tl.constexpr,  # 128
    H: tl.constexpr,
    W: tl.constexpr,
    # Stride for softmax (after reshape to [batch, 2, 128])
    softmax_stride_0: tl.constexpr,
    softmax_stride_1: tl.constexpr,
    softmax_stride_2: tl.constexpr,
    # Stride for weight (contiguous [batch, 2, num_positions, H, W])
    weight_stride_0: tl.constexpr,
    weight_stride_1: tl.constexpr,
    weight_stride_2: tl.constexpr,
    weight_stride_3: tl.constexpr,
    weight_stride_4: tl.constexpr,
):
    """
    Fused kernel for: softmax -> multiply -> sum -> contiguous
    
    For each batch and each output position (pos, h, w):
    - Compute softmax on the (reshaped) input along dim=1 (2 elements)
    - Weighted sum: s0 * w0 + s1 * w1
    
    Input:
    - softmax_ptr after reshape: [batch, 2, num_positions=128]
    - weight_ptr: [batch, 2, num_positions, H, W]
    
    Output:
    - out: [batch, num_positions, H, W]
    """
    # Get program position
    batch_idx = tl.program_id(0)
    out_idx = tl.program_id(1)
    
    if batch_idx >= batch:
        return
    
    # Compute position indices
    # out_idx = pos * H * W + h * W + w
    spatial_per_pos = H * W
    pos_idx = out_idx // spatial_per_pos
    spatial_idx = out_idx % spatial_per_pos
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W
    
    # Load softmax values for this batch and position
    # softmax after reshape: [batch, 2, num_positions]
    softmax_base = batch_idx * softmax_stride_0 + pos_idx
    s0 = tl.load(softmax_ptr + softmax_base)
    s1 = tl.load(softmax_ptr + softmax_base + softmax_stride_1)
    
    # Load weight values
    # weight: [batch, 2, num_positions, H, W]
    weight_base = (
        batch_idx * weight_stride_0
        + pos_idx * weight_stride_2
        + h_idx * weight_stride_3
        + w_idx * weight_stride_4
    )
    w0 = tl.load(weight_ptr + weight_base)
    w1 = tl.load(weight_ptr + weight_base + weight_stride_1)
    
    # Compute weighted sum
    result = s0 * w0 + s1 * w1
    
    # Store result: [batch, num_positions, H, W]
    out_offset = batch_idx * num_positions * spatial_per_pos + out_idx
    tl.store(out_ptr + out_offset, result)


@torch.fx.wrap
def fused_softmax_weighted_sum(in_0, in_1, route):
    """
    Fused kernel: softmax + reshape + multiply + sum + contiguous
    
    Args:
        in_0: weight tensor with shape [batch, 2, num_positions, H, W]
        in_1: attention tensor with shape [batch, 2, 1, num_positions]
    Returns:
        output with shape [batch, num_positions, H, W]
    """
    batch_size = in_0.shape[0]
    num_positions = in_0.shape[2]  # 128
    H = in_0.shape[3]
    W = in_0.shape[4]
    
    # Output shape: [batch, num_positions, H, W]
    output_shape = (batch_size, num_positions, H, W)
    out = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Reshape in_1: [batch, 2, 1, 128] -> [batch, 2, 128]
    in_1_reshaped = in_1.reshape(batch_size, 2, num_positions)
    
    # Grid: (batch, num_positions * H * W)
    total_out_per_batch = num_positions * H * W
    grid = (batch_size, total_out_per_batch)
    
    fused_softmax_weighted_sum_kernel[grid](
        softmax_ptr=in_1_reshaped,
        weight_ptr=in_0,
        out_ptr=out,
        batch=batch_size,
        num_positions=num_positions,
        H=H,
        W=W,
        softmax_stride_0=2 * num_positions,
        softmax_stride_1=num_positions,
        softmax_stride_2=1,
        weight_stride_0=2 * num_positions * H * W,
        weight_stride_1=num_positions * H * W,
        weight_stride_2=H * W,
        weight_stride_3=W,
        weight_stride_4=1,
    )
    
    return out.contiguous()


def pattern(in_0, in_1):
    """
    Match the fused pattern:
    softmax(in_1, dim=1) -> reshape -> view -> view -> multiply -> sum -> contiguous
    """
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(-1, 2, 1, 128)
    tmp_2 = tmp_1.view(-1, 2, 1, 1, 1)
    tmp_3 = tmp_2.view(-1, 2, 1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the fused kernel.
    Route string for dispatch when output_pass_replacement_func_limit=1.
    """
    return (in_0, in_1, "softmax_weighted_sum_route")


def replacement_func():
    return fused_softmax_weighted_sum