import torch
import triton
import triton.language as tl

def conv2d_reshape_permute(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups, reshape_shape, permute_dims):
    """Pattern function to match Conv2D + Reshape + Permute sequence"""
    tmp_4 = torch.conv2d(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups)
    tmp_5 = tmp_4.reshape(*reshape_shape)
    tmp_6 = tmp_5.permute(*permute_dims)
    return tmp_6

def replacement_args(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups, reshape_shape, permute_dims):
    return (input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups, reshape_shape, permute_dims)

@triton.jit
def fused_conv2d_reshape_permute_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_channels,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate output dimensions
    out_height = (in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) // stride_w + 1
    
    # Reshape and permute output dimensions
    reshape_dim = out_height * out_width
    permute_batch = batch_size
    permute_seq = reshape_dim
    permute_channels = out_channels
    
    # Grid setup for the permuted layout
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1) 
    channel_idx = tl.program_id(2)
    
    # Calculate original conv2d output indices
    row_idx = seq_idx // out_width
    col_idx = seq_idx % out_width
    
    # Conv2D computation with fused output layout
    for m in range(0, out_channels, BLOCK_SIZE):
        channels = tl.arange(m, min(m + BLOCK_SIZE, out_channels))
        
        # Load and compute convolution
        val = tl.zeros([channels.shape[0]], dtype=tl.float32)
        
        # Compute output for this block of channels
        for c in range(channels.shape[0]):
            channel = channels[c]
            group = channel // (out_channels // groups)
            
            # Loop over kernel
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input position
                    in_row = row_idx * stride_h + kh * dilation_h - padding_h
                    in_col = col_idx * stride_w + kw * dilation_w - padding_w
                    
                    if 0 <= in_row < in_height and 0 <= in_col < in_width:
                        # Load input patch
                        base_offset = (batch_idx * in_channels + group * (in_channels // groups)) * in_height * in_width
                        patch_offset = (in_row * in_width + in_col) * (in_channels // groups)
                        input_offset = base_offset + patch_offset + (channel % (in_channels // groups))
                        
                        input_val = tl.load(input_ptr + input_offset)
                        weight_offset = channel * kernel_height * kernel_width + kh * kernel_width + kw
                        weight_val = tl.load(weight_ptr + weight_offset)
                        
                        val[c] += input_val * weight_val
            
            # Add bias
            bias_val = tl.load(bias_ptr + channel)
            val[c] += bias_val
        
        # Directly store in permuted layout [batch, seq_len, channels]
        permuted_offset = (batch_idx * permute_seq * permute_channels + 
                          seq_idx * permute_channels + 
                          channel_idx)
        if channel_idx < channels.shape[0]:
            tl.store(out_ptr + permuted_offset, val[channel_idx])

@torch.fx.wrap
def fused_conv2d_reshape_permute(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups, reshape_shape, permute_dims):
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight_tensor.shape
    
    # Determine output dimensions
    out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    reshape_dim = out_height * out_width
    
    # Output shape is [batch_size, reshape_dim, out_channels]
    output_shape = [batch_size, reshape_dim, out_channels]
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block size for channel parallelization
    BLOCK_SIZE = 64
    
    # Grid setup: [batch, sequence length, channels]
    grid = (
        batch_size,
        reshape_dim,
        (out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )
    
    fused_conv2d_reshape_permute_kernel[grid](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output_tensor,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        BLOCK_SIZE,
    )
    
    return output_tensor

def replacement_func():
    return fused_conv2d_reshape_permute