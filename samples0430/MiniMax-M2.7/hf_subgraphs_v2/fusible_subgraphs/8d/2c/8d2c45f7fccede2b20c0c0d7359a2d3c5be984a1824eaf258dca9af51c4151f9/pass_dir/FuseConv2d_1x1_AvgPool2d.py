import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv_pool_kernel(
    # Convolution inputs
    x_ptr, weight_ptr, out_ptr,
    # Conv params
    IN_C, IN_H, IN_W, OUT_H, OUT_W, OUT_C,
    # Strides
    x_batch_stride, x_channel_stride, x_h_stride, x_w_stride,
    w_out_channel_stride,
    out_batch_stride, out_channel_stride, out_h_stride, out_w_stride,
):
    """
    Fused kernel: 1x1 Conv2d + AvgPool2d(stride=2)
    
    Optimized for each program to handle one output position (all channels).
    Uses a blocked approach over output channels with loop over input channels.
    
    Grid: (batch * OUT_H * OUT_W)
    Each program computes all OUT_C channels for one output position.
    """
    # Program index
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    
    # Pool window in input
    in_h_base = out_h_idx * 2
    in_w_base = out_w_idx * 2
    
    # Output offset base for this batch/spatial position
    out_offset_base = (batch_idx * out_batch_stride + 
                       out_h_idx * out_h_stride + 
                       out_w_idx * out_w_stride)
    
    # Block size for processing output channels
    BLOCK_SIZE = 64  # Process 64 output channels at a time
    
    # Number of channel blocks
    num_channel_blocks = (OUT_C + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Offsets for channel indexing
    offs_c = tl.arange(0, BLOCK_SIZE)
    offs_ic = tl.arange(0, IN_C)
    
    for ch_block_idx in range(num_channel_blocks):
        ch_start = ch_block_idx * BLOCK_SIZE
        ch_end = min(ch_start + BLOCK_SIZE, OUT_C)
        block_len = ch_end - ch_start
        
        # Mask for valid channels in this block
        c_mask = offs_c < block_len
        
        # Initialize accumulator (float32 for precision)
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Load weights for this block: shape [block_len, IN_C]
        # Weight layout: [OUT_C, IN_C, 1, 1], so weight[oc, ic] = weight_ptr[oc * IN_C + ic]
        # For channels [ch_start:ch_end], we want:
        # w_block[oc_local, ic] = weight_ptr[(ch_start + oc_local) * IN_C + ic]
        w_base = ch_start * IN_C
        w_offsets = (offs_c * IN_C)[:, None] + offs_ic[None, :]
        w_block = tl.load(weight_ptr + w_base + tl.reshape(w_offsets, [BLOCK_SIZE * IN_C]),
                          mask=c_mask[:, None], other=0.0)
        w_block = tl.reshape(w_block, [BLOCK_SIZE, IN_C])
        
        # For pooling: need to sum 4 input pixels
        # Each input pixel (b, ic, ih, iw) contributes to output for all output channels
        # inp_val[ic] = sum of 4 input pixels for this ic
        
        # Initialize pooled input accumulator
        inp_pooled = tl.zeros([BLOCK_SIZE, IN_C], dtype=tl.float32)
        
        # Process each pool position
        for ph in range(2):
            for pw in range(2):
                in_h = in_h_base + ph
                in_w = in_w_base + pw
                
                # Check bounds
                if in_h < IN_H and in_w < IN_W:
                    # Load all input channels at this position
                    # x[b, ic, in_h, in_w]
                    x_base = batch_idx * x_batch_stride + in_h * x_h_stride + in_w * x_w_stride
                    x_offsets = x_base + offs_ic * x_channel_stride
                    x_vals = tl.load(x_ptr + x_offsets, mask=offs_ic < IN_C, other=0.0)
                    
                    # Broadcast x_vals (shape [IN_C]) to shape [BLOCK_SIZE, IN_C]
                    # For each row (output channel), same x_vals apply
                    inp_pooled = inp_pooled + x_vals[None, :]
        
        # Now compute: acc[oc] = sum_ic inp_pooled[oc, ic] * w_block[oc, ic]
        # This is row-wise dot product
        for ic_idx in range(IN_C):
            acc = acc + inp_pooled[:, ic_idx] * w_block[:, ic_idx]
        
        # Divide by pool area (4)
        acc = acc / 4.0
        
        # Store results for this block
        for oc_local in range(block_len):
            oc = ch_start + oc_local
            out_offset = out_offset_base + oc * out_channel_stride
            tl.store(out_ptr + out_offset, acc[oc_local])


@torch.fx.wrap
def fused_conv_pool_wrapper(x, weight, IN_C, IN_H, IN_W, OUT_C):
    """Wrapper for the fused conv + pool kernel"""
    batch, _, _, _ = x.shape
    
    OUT_H = (IN_H + 1) // 2
    OUT_W = (IN_W + 1) // 2
    
    # Allocate output with same dtype as input
    out = torch.empty((batch, OUT_C, OUT_H, OUT_W), 
                      dtype=x.dtype, device=x.device)
    
    # Get strides
    x_batch_stride, x_channel_stride, x_h_stride, x_w_stride = x.stride()
    w_out_channel_stride = weight.stride(0)
    out_batch_stride, out_channel_stride, out_h_stride, out_w_stride = out.stride()
    
    # Launch kernel: (batch, OUT_H, OUT_W)
    grid = (batch, OUT_H, OUT_W)
    
    fused_conv_pool_kernel[grid](
        x, weight, out,
        IN_C, IN_H, IN_W, OUT_H, OUT_W, OUT_C,
        x_batch_stride, x_channel_stride, x_h_stride, x_w_stride,
        w_out_channel_stride,
        out_batch_stride, out_channel_stride, out_h_stride, out_w_stride,
    )
    
    return out


# Module-level function for replacement
def fused_conv_pool(in_0, in_1, IN_C, IN_H, IN_W, OUT_C):
    """Fused conv + pool operation"""
    return fused_conv_pool_wrapper(in_1, in_0, IN_C, IN_H, IN_W, OUT_C)


def pattern(in_0, in_1):
    """
    Match: Conv2d(1x1, stride=1, padding=0) -> AvgPool2d(kernel=2, stride=2, padding=0)
    
    Args:
        in_0: weight tensor [out_c, in_c, 1, 1]
        in_1: input tensor [batch, in_c, H, W]
    
    Returns:
        pooled output tensor
    """
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    return tmp_2


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the fused kernel.
    
    Conv2d args:
        in_1: input [batch, in_c, H, W]
        in_0: weight [out_c, in_c, 1, 1]
    
    Pooling doesn't need additional params (fixed kernel=2, stride=2)
    """
    batch, IN_C, IN_H, IN_W = in_1.shape
    OUT_C = in_0.shape[0]
    
    return (in_0, in_1, IN_C, IN_H, IN_W, OUT_C)


def replacement_func():
    """
    Returns the fused conv+pool function.
    
    The fused kernel performs:
    1. 1x1 Conv2d (stride=1, padding=0, dilation=1, groups=1)
    2. AvgPool2d (kernel=2, stride=2, padding=0)
    
    Fusing these operations:
    - Reduces global memory traffic by avoiding intermediate writes
    - Improves cache utilization
    - Allows better instruction-level parallelism
    """
    return fused_conv_pool