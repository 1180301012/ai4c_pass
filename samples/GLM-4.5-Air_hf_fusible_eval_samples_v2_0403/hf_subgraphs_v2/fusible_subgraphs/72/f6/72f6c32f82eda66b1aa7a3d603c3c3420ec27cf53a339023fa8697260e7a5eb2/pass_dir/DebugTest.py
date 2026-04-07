import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def triton_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    batch_id = pid // (height * width)
    spatial_pid = pid % (height * width)
    h = spatial_pid // width
    w = spatial_pid % width
    
    # For 1x1 convolution: sum over input channels for each position
    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    for k in range(0, in_channels, BLOCK_SIZE_K):
        channels_block = min(BLOCK_SIZE_K, in_channels - k)
        
        # Load input element
        input_offset = (batch_id * height + h) * width + w + k
        input_vals = tl.load(input_ptr + input_offset, mask=k < in_channels, other=0.0).to(tl.float32)
        
        # Load weight for output channels
        weight_ptrs = weight_ptr + k * out_channels
        weight_vals = tl.load(weight_ptrs, mask=tl.arange(0, channels_block) < out_channels, other=0.0)
        
        # Matrix multiplication (input channels x output channels)
        for c_out in range(BLOCK_SIZE_N):
            if c_out < out_channels:
                acc[c_out] += tl.sum(input_vals * weight_vals[c_out])
    
    # Store result
    for c_out in range(0, out_channels, BLOCK_SIZE_N):
        channels_block = min(BLOCK_SIZE_N, out_channels - c_out)
        if channels_block > 0:
            out_offset = ((batch_id * height + h) * width + w) * out_channels + c_out
            tl.store(output_ptr + out_offset + acc[c_out:c_out+channels_block], c_out < out_channels)

@torch.fx.wrap
def triton_conv2d(input_tensor, weight_tensor):
    batch, in_channels, height, width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight_tensor.shape
    
    assert kernel_h == 1 and kernel_w == 1, "Only 1x1 convolution supported"
    
    output_shape = (batch, out_channels, height, width)
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    spatial_elements = height * width
    total_elements = batch * spatial_elements
    grid_size = (triton.cdiv(total_elements, BLOCK_SIZE_M),)
    
    triton_conv2d_kernel[grid_size](
        input_tensor,
        weight_tensor,
        output_tensor,
        batch,
        in_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return output_tensor

def replacement_func():
    return triton_conv2d