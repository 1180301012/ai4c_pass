import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Pattern to match: softmax + reshape + view + multiply + sum (batch_size=1)"""
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(1, -1)
    tmp_2 = tmp_1.view(1, -1, 1, 1)
    tmp_3 = tmp_2.view(1, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return (tmp_6,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_HW': 128}),
        triton.Config({'BLOCK_SIZE_HW': 256}),
        triton.Config({'BLOCK_SIZE_HW': 512}),
        triton.Config({'BLOCK_SIZE_HW': 1024}),
        triton.Config({'BLOCK_SIZE_HW': 2048}),
    ],
    key=['hw_total'],
)
@triton.jit
def fused_split_attention_kernel_b1(
    in_0_ptr, in_1_ptr, out_ptr,
    batch_size, num_splits, num_channels, height, width, hw_total,
    stride_in0_b, stride_in0_s, stride_in0_c, stride_in0_h, stride_in0_w,
    stride_in1_b, stride_in1_s, stride_in1_0, stride_in1_c,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE_HW: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. Softmax on in_1 along dim=1 (num_splits=2)
    2. Weighted sum of in_0 using softmax weights
    
    Grid: (batch_size, num_channels, num_hw_blocks)
    """
    # Get program IDs for 3D grid
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    b_idx = pid_b
    c_idx = pid_c
    
    # Load attention logits for both splits: in_1[b, :, 0, c]
    offset_s0 = b_idx * stride_in1_b + 0 * stride_in1_s + c_idx * stride_in1_c
    offset_s1 = b_idx * stride_in1_b + 1 * stride_in1_s + c_idx * stride_in1_c
    logit_s0 = tl.load(in_1_ptr + offset_s0)
    logit_s1 = tl.load(in_1_ptr + offset_s1)
    
    # Compute softmax (stable version)
    max_logit = tl.maximum(logit_s0, logit_s1)
    exp_s0 = tl.exp(logit_s0 - max_logit)
    exp_s1 = tl.exp(logit_s1 - max_logit)
    exp_sum = exp_s0 + exp_s1
    
    att_weight_s0 = exp_s0 / exp_sum
    att_weight_s1 = exp_s1 / exp_sum
    
    # Process this block of H*W positions
    hw_start = pid_hw * BLOCK_SIZE_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE_HW)
    hw_mask = hw_offsets < hw_total
    
    # Decompose linear hw index into (h, w)
    w_indices = hw_offsets % width
    h_indices = hw_offsets // width
    
    # Load values from both splits: in_0[b, :, c, h, w]
    offset_in0_s0 = (b_idx * stride_in0_b + 0 * stride_in0_s + 
                     c_idx * stride_in0_c + h_indices * stride_in0_h + w_indices * stride_in0_w)
    offset_in0_s1 = (b_idx * stride_in0_b + 1 * stride_in0_s + 
                     c_idx * stride_in0_c + h_indices * stride_in0_h + w_indices * stride_in0_w)
    
    values_s0 = tl.load(in_0_ptr + offset_in0_s0, mask=hw_mask, other=0.0)
    values_s1 = tl.load(in_0_ptr + offset_in0_s1, mask=hw_mask, other=0.0)
    
    # Compute weighted sum (attention-weighted combination)
    result = att_weight_s0 * values_s0 + att_weight_s1 * values_s1
    
    # Store result: out[b, c, h, w]
    offset_out = b_idx * stride_out_b + c_idx * stride_out_c + h_indices * stride_out_h + w_indices * stride_out_w
    tl.store(out_ptr + offset_out, result, mask=hw_mask)


@torch.fx.wrap
def fused_split_attention_b1(in_0, in_1):
    """
    Wrapper function for the fused split-attention kernel.
    
    Args:
        in_0: Input tensor [B, 2, C, H, W]
        in_1: Attention logits [B, 2, 1, C]
    
    Returns:
        Output tensor [B, C, H, W]
    """
    batch_size, num_splits, num_channels, height, width = in_0.shape
    hw_total = height * width
    
    # Output shape: [batch_size, num_channels, height, width]
    out = torch.empty((batch_size, num_channels, height, width), 
                      device=in_0.device, dtype=in_0.dtype)
    
    # Determine grid size
    # We'll use a placeholder BLOCK_SIZE_HW for calculating num_hw_blocks
    # The autotune will override this
    BLOCK_SIZE_HW = 1024
    num_hw_blocks = triton.cdiv(hw_total, BLOCK_SIZE_HW)
    
    # 3D grid: (batch_size, num_channels, num_hw_blocks)
    grid = lambda meta: (batch_size, num_channels, triton.cdiv(hw_total, meta['BLOCK_SIZE_HW']))
    
    fused_split_attention_kernel_b1[grid](
        in_0, in_1, out,
        batch_size, num_splits, num_channels, height, width, hw_total,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3), in_0.stride(4),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    
    return out


def replacement_func():
    return fused_split_attention_b1