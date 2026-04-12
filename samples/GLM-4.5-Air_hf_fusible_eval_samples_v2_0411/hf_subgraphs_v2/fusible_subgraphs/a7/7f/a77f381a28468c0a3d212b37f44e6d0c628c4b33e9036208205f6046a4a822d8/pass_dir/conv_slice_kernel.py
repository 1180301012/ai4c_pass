import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_channel_select_kernel(
    input_ptr,      # [N, C_in, H_in, W_in]
    weight_ptr,     # [C_out, C_in, H_k, W_k] 
    output_ptr,     # [N, C_out_selected, H_out, W_out]
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    channel_start, channel_end,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    # Calculate output coordinates
    n_offset = tl.program_id(0) * BLOCK_SIZE_N
    h_offset = tl.program_id(1) * BLOCK_SIZE_H
    
    n = n_offset + tl.arange(0, BLOCK_SIZE_N)
    h = h_offset + tl.arange(0, BLOCK_SIZE_H)
    w = tl.arange(0, W_out)
    c_out = tl.arange(0, BLOCK_SIZE_C)
    
    # Create masks for bounds checking
    n_mask = n < N
    h_mask = h < H_out
    w_mask = w < W_out
    c_out_mask = c_out < (channel_end - channel_start)
    
    # Only process channels in the selected range
    c_out_abs = c_out + channel_start
    c_out_abs_mask = c_out_abs < C_out
    
    # Compute effective indices
    h_in = h * stride_h - pad_h
    w_in = w * stride_w - pad_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, W_out), dtype=tl.float32)
    
    # Loop over input channels and kernel sizes
    for c_in_idx in range(0, C_in, BLOCK_SIZE_C):
        c_in = c_in_idx + tl.arange(0, BLOCK_SIZE_C)
        c_in_mask = c_in < C_in
        
        # Load input patch [N, C_in, H_out, W_out]
        input_ptrs = input_ptr + (n[:, None, None, None] * C_in * H_in * W_in +
                                 c_in[None, :, None, None] * H_in * W_in +
                                 (h_in[:, None, :, None] * W_in + w_in[None, None, :, None]))
        input_patch = tl.load(input_ptrs, mask=n_mask[:, None, None, None] & c_in_mask[None, :, None, None] & h_mask[:, None, None, None] & w_mask[None, None, :, None], other=0.0).to(tl.float32)
        
        # Load kernel weights [C_out, C_in, H_k, W_k]
        weight_ptrs = weight_ptr + (c_out_abs[:, None, None, None] * C_in * 1 * 1 +
                                    c_in[None, :, None, None] * 1 * 1 +
                                    tl.arange(0, 1)[None, None, :, None] * 1 +
                                    tl.arange(0, 1)[None, None, None, :])
        weight_patch = tl.load(weight_ptrs, mask=c_out_abs_mask[:, None, None, None] & c_in_mask[None, :, None, None], other=0.0).to(tl.float32)
        
        # Convolution operation
        acc += tl.sum(input_patch[:, :, :, :, None, None] * weight_patch[None, :, None, None, :, :], axis=2)
    
    # Store output [N, C_out_selected, H_out, W_out]
    output_ptrs = output_ptr + (n[:, None, None, None] * (channel_end - channel_start) * H_out * W_out +
                                c_out[None, :, None, None] * H_out * W_out +
                                h[:, None, :, None] * W_out + w[None, None, :, None])
    
    tl.store(output_ptrs, acc.to(tl.float16), mask=n_mask[:, None, None, None] & c_out_mask[None, :, None, None] & h_mask[:, None, None, None] & w_mask[None, None, :, None])

@torch.fx.wrap
def conv2d_channel_select(input, weight, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, channel_start=0, channel_end=None):
    if channel_end is None:
        channel_end = input.size(0)
    
    N, C_in, H_in, W_in = input.shape
    C_out, _, H_k, W_k = weight.shape
    
    # Calculate output dimensions
    H_out = (H_in + 2 * padding[0] - dilation[0] * (H_k - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - dilation[1] * (W_k - 1) - 1) // stride[1] + 1
    
    selected_channels = channel_end - channel_start
    
    # Allocate output
    output = torch.empty((N, selected_channels, H_out, W_out), dtype=input.dtype, device=input.device)
    
    # Launch kernel
    BLOCK_SIZE_N = 4
    BLOCK_SIZE_C = 64
    BLOCK_SIZE_H = 16
    
    n_blocks = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    h_blocks = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    
    conv2d_channel_select_kernel[(n_blocks, h_blocks, 1)](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        N=N, C_in=C_in, H_in=H_in, W_in=W_in,
        C_out=C_out, H_out=H_out, W_out=W_out,
        channel_start=channel_start, channel_end=channel_end,
        stride_h=stride[0], stride_w=stride[1],
        pad_h=padding[0], pad_w=padding[1],
        dilation_h=dilation[0], dilation_w=dilation[1],
        groups=groups,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    
    return output

@torch.fx.wrap  
def conv2d_full_and_slice(input, weight, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, channel_slice=slice(None, 2048, None)):
    # Full convolution
    full_conv = torch.conv2d(input, weight, None, stride, padding, dilation, groups)
    
    # Channel slicing
    sliced_conv = full_conv[(slice(None, None, None), channel_slice, slice(None, None, None), slice(None, None, None))]
    
    return full_conv, sliced_conv