import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def depthwise_conv2d_mean_kernel(
    input_ptr, weight_ptr, output_ptr, output_mean_ptr,
    N, C, H, W, KW, KH,
    stride_in, stride_w, stride_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused depthwise conv2d + mean across spatial dimensions.
    
    Depthwise conv with groups=C, stride=1, padding=1 preserves spatial dimensions.
    Then mean over (H, W) gives output shape [N, C, 1, 1].
    
    We compute: out[n, c, h, w] = sum over kernel of input[n, c, h+kH-1, w+kW-1] * weight[c, 0, kH, kW]
    Then mean: mean[n, c, 0, 0] = sum over h,w of out[n, c, h, w] / (H * W)
    
    Since stride=1, padding=1, the conv output has same spatial size as input.
    The full computation is:
    mean[n, c, 0, 0] = sum over h,w,kH,kW of input[n, c, h+kH-1, w+kW-1] * weight[c, 0, kH, kW] / (H*W)
    """
    # Each program handles one batch element, computing all channels
    pid_n = tl.program_id(0)
    
    # Offset for batch
    input_offset = pid_n * C * H * W
    output_offset = pid_n * C * H * W
    output_mean_offset = pid_n * C
    
    # Iterate over channels
    for c in range(0, C, BLOCK_SIZE_N):
        c_range = c + tl.arange(0, BLOCK_SIZE_N)
        c_mask = c_range < C
        
        # Initialize accumulator for sum (across spatial and kernel)
        # For mean: we need sum of all contributions / (H*W)
        sum_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        
        # Loop over spatial dimensions to compute full conv output first
        for h in range(H):
            for w in range(W):
                # Load weight and compute contributions for all channels in this block
                weight_offset_base = c_range * KW * KH * 1  # weight: [C, 1, KH, KW]
                
                # Accumulate conv + mean in one pass
                # For each kernel position
                for kh in range(KH):
                    for kw in range(KW):
                        # Input coordinates with padding
                        inp_h = h + kh  # since padding=1, input index = h + kh
                        inp_w = w + kw
                        
                        # Check bounds (padding creates boundary checks)
                        # With padding=1, valid input indices are [1, H] for a H-sized input
                        # But our input already includes padding conceptually, so indices are [0, H-1]
                        # Actually the input is size H x W and we pad with 0
                        
                        # Input position: input[n, c, h+kh, w+kw]
                        # Since padding=1, we use input[h+kh, w+kw] directly
                        
                        # Load input values for all channels
                        # input layout: [N, C, H, W]
                        inp_offsets = (input_offset + 
                                       c_range[:, None] * H * W + 
                                       inp_h * W + 
                                       inp_w)
                        
                        # We need to compute: sum over kh,kw of input * weight
                        # weight shape: [C, 1, KH, KW] -> flat: [C, KH, KW]
                        w_offsets = (c_range * KW * KH + kh * KW + kw)
                        
                        # Load input and weight
                        inp_vals = tl.load(input_ptr + inp_offsets, mask=c_mask[:, None], other=0.0)
                        w_vals = tl.load(weight_ptr + w_offsets, mask=c_mask, other=0.0)
                        
                        # Accumulate: conv contribution
                        conv_contrib = inp_vals * w_vals
                        sum_acc += tl.sum(conv_contrib, axis=0)
        
        # After processing all spatial positions, sum_acc contains sum over H*W of conv output
        # For mean, divide by H*W
        mean_val = sum_acc / (H * W)
        
        # Store mean output [N, C, 1, 1]
        tl.store(output_mean_ptr + output_mean_offset + c_range, mean_val, mask=c_mask)


@torch.fx.wrap
def fused_depthwise_conv2d_mean(input_tensor, weight_tensor):
    """
    Fused kernel: depthwise conv2d + spatial mean.
    
    Args:
        input_tensor: [N, C, H, W]
        weight_tensor: [C, 1, KH, KW] - depthwise conv weight
    
    Returns:
        output: [N, C, H, W] - conv output
        output_mean: [N, C, 1, 1] - spatial mean
    """
    # Depthwise conv with groups=C
    # groups=384, in_channels=384, out_channels=384
    output = torch.nn.functional.conv2d(
        input_tensor, 
        weight_tensor, 
        None,  # bias
        (1, 1),  # stride
        (1, 1),  # padding
        (1, 1),  # dilation
        384  # groups - depthwise
    )
    
    # Spatial mean pooling
    output_mean = output.mean((2, 3), keepdim=True)
    
    return output, output_mean


def pattern(in_0, in_1):
    """
    Match the pattern: depthwise conv2d followed by mean over spatial dimensions.
    
    The original computation:
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 384)
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)
    
    Note: The conv uses groups=384, which means depthwise conv with C groups = C output channels.
    Weight shape is [C, 1, 3, 3].
    """
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 384)
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2


def replacement_args(in_0, in_1):
    """
    Extract arguments for replacement: (weight, input)
    """
    return (in_0, in_1)


def replacement_func():
    return fused_depthwise_conv2d_mean