import torch
import triton
import triton.language as tl

@triton.jit
def fuse_se_block_relu_avgpool_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, out_ptr,
    stride_in_0, stride_in_1, stride_in_2,
    stride_out,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SE Block + ReLU + AdaptiveAvgPool2d kernel
    
    Operations fused:
    1. sigmoid(in_2) -> view(1, -1, 1, 1) -> expand -> multiply -> add -> relu
    2. adaptive_avg_pool2d(output, 1) -> flatten
    
    Inputs:
    - in_0: residual [1, C, H, W]
    - in_1: features [1, C, H, W] 
    - in_2: squeeze [1, 1, C]
    
    Output:
    - out: pooled features [1, C]
    """
    pid = tl.program_id(0)
    
    # Calculate number of channels to process
    num_channels = C
    
    # Each program processes a subset of channels
    channels_per_block = BLOCK_SIZE
    start_ch = pid * channels_per_block
    mask_ch = start_ch + tl.arange(0, channels_per_block) < num_channels
    
    # Load in_2 (squeeze output) - shape [1, 1, C]
    # Position: in_2[b, 0, c] = in_2_ptr[0 * stride_in_2[0] + 0 * stride_in_2[1] + c * stride_in_2[2]]
    squeeze = tl.load(in_2_ptr + start_ch * stride_in_2[2], mask=mask_ch, other=0.0)
    
    # Compute sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_squeeze = 1.0 / (1.0 + tl.exp(-squeeze))
    
    # Store sigmoid result for reuse in the loop
    sigmoid_local = sigmoid_squeeze
    
    # For each channel, compute:
    # 1. sigmoid_squeeze[c] * in_1[0, c, h, w]
    # 2. + in_0[0, c, h, w]
    # 3. relu(result)
    # 4. accumulate into avg_pool accumulator
    
    # Initialize accumulators for adaptive_avg_pool2d (summing over H*W spatial positions)
    acc = tl.zeros((channels_per_block,), dtype=tl.float32)
    
    # Loop over spatial positions
    for h in range(H):
        for w in range(W):
            # Compute linear offsets for in_0, in_1
            # in_0: [b, c, h, w] -> b*stride[0] + c*stride[1] + h*stride[2] + w*stride[3]
            offset_0 = h * stride_in_0[2] + w * stride_in_0[3]
            offset_1 = h * stride_in_1[2] + w * stride_in_1[3]
            
            # Load in_0 and in_1 for current channel and spatial position
            val_0 = tl.load(in_0_ptr + start_ch * stride_in_0[1] + offset_0, mask=mask_ch, other=0.0)
            val_1 = tl.load(in_1_ptr + start_ch * stride_in_1[1] + offset_1, mask=mask_ch, other=0.0)
            
            # Compute: sigmoid_squeeze[c] * val_1 + val_0
            scaled = sigmoid_local * val_1
            summed = scaled + val_0
            
            # ReLU: max(0, x)
            relu_out = tl.where(summed > 0, summed, 0.0)
            
            # Accumulate for average pooling
            acc = acc + relu_out
    
    # Compute average: divide by H * W
    avg_pool = acc / (H * W)
    
    # Clamp to prevent extreme values (numerical stability)
    # avg_pool = tl.clamp(avg_pool, -80.0, 80.0)
    
    # Store output: [1, C] - each program stores its channels
    out_offset = start_ch * stride_out[1]
    tl.store(out_ptr + out_offset, avg_pool, mask=mask_ch)


@torch.fx.wrap
def fuse_se_block_relu_avgpool_wrapper(in_0, in_1, in_2):
    """
    Wrapper for the fused SE Block + ReLU + AdaptiveAvgPool2d kernel.
    
    Args:
        in_0: residual tensor [1, C, H, W]
        in_1: features tensor [1, C, H, W]
        in_2: squeeze tensor [1, 1, C]
    
    Returns:
        Pooled output [1, C]
    """
    N, C, H, W = in_1.shape
    
    # Allocate output tensor
    out = torch.empty((N, C), dtype=in_1.dtype, device=in_1.device)
    
    # Get strides
    stride_in_0 = in_0.stride()
    stride_in_1 = in_1.stride()
    stride_in_2 = in_2.stride()
    stride_out = out.stride()
    
    # Determine block size based on number of channels
    # C is typically 2048 for ResNet models
    BLOCK_SIZE = min(1024, triton.next_power_of_2(C))
    
    # Calculate grid size
    num_programs = (C + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fuse_se_block_relu_avgpool_kernel[(num_programs,)](
        in_0, in_1, in_2, out,
        stride_in_0, stride_in_1, stride_in_2,
        stride_out,
        N, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1, in_2):
    """
    Match the pattern exactly as written in model.py
    """
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    tmp_3 += in_0
    tmp_4 = tmp_3
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fuse_se_block_relu_avgpool_wrapper