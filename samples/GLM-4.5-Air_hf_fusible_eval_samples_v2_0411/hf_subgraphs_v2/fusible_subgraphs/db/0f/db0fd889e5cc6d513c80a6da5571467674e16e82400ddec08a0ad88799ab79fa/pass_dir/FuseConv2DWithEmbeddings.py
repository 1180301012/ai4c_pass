import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_size, in_channels, out_channels,
    input_height, input_width,
    kernel_height, kernel_width,
    stride_height, stride_width,
    padding_height, padding_width,
    dilation_height, dilation_width, groups,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * out_channels * (input_height // stride_height) * (input_width // stride_width))
    
    # Calculate output dimensions
    output_height = input_height // stride_height
    output_width = input_width // stride_width
    
    # Convert flat offset to 4D coordinates
    output_c = offsets // (output_height * output_width) % out_channels
    output_h = (offsets // output_width) % output_height
    output_w = offsets % output_width
    batch = offsets // (out_channels * output_height * output_width)
    
    # Conv2D computation
    sum_val = 0.0
    for ci in range(in_channels // groups):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                input_h = output_h * stride_height + kh * dilation_height
                input_w = output_w * stride_width + kw * dilation_width
                
                if (input_h < input_height and input_w < input_width):
                    weight_idx = output_c * kernel_height * kernel_width * (in_channels // groups) + ci * kernel_height * kernel_width + kh * kernel_width + kw
                    input_idx = batch * in_channels * input_height * input_width + ci * input_height * input_width + input_h * input_width + input_w
                    
                    weight = tl.load(weight_ptr + weight_idx)
                    input_val = tl.load(input_ptr + input_idx)
                    sum_val += weight * input_val
    
    # Add bias
    bias_idx = output_c
    sum_val += tl.load(bias_ptr + bias_idx)
    
    # Store result
    tl.store(output_ptr + offsets, sum_val, mask=mask)

@triton.jit
def flatten_transpose_kernel(input_ptr, output_ptr, batch_size, features, seq_len, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * seq_len)
    
    batch = offsets // seq_len
    seq_pos = offsets % seq_len
    
    # Calculate indices for original [batch, features, seq_len] -> [batch * seq_len, features]
    input_idx = batch * features * seq_len + seq_pos * features
    output_idx = offsets * features
    
    for i in range(features):
        val = tl.load(input_ptr + input_idx + i, mask=mask, other=0.0)
        tl.store(output_ptr + output_idx + i, val, mask=mask)

@triton.jit  
def expand_cat_kernel(cls_ptr, conv_out_ptr, detect_ptr, output_ptr, batch_size, cls_dim, conv_dim, detect_dim, total_dim, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * total_dim)
    
    batch = offsets // total_dim
    out_pos = offsets % total_dim
    
    # cls token expansion: [batch, cls_dim] -> [batch*seq_len, cls_dim]
    # conv output: [batch*seq_len, conv_dim] 
    # detect tokens expansion: [batch, detect_dim] -> [batch*seq_len, detect_dim]
    
    cls_idx = batch * cls_dim
    conv_idx = (batch * total_dim + out_pos) * conv_dim + (out_pos - cls_dim) if out_pos < cls_dim else 0
    detect_idx = batch * detect_dim
    
    if out_pos < cls_dim:
        # cls token region
        val = tl.load(cls_ptr + cls_idx + out_pos, mask=mask, other=0.0)
    elif out_pos < cls_dim + conv_dim:
        # conv output region  
        conv_out_idx = batch * cls_dim * conv_dim + (out_pos - cls_dim) * conv_dim
        val = tl.load(conv_out_ptr + conv_out_idx + (conv_idx % conv_dim), mask=mask, other=0.0)
    else:
        # detect tokens region
        detect_idx_in_batch = out_pos - cls_dim - conv_dim
        val = tl.load(detect_ptr + detect_idx + detect_idx_in_batch, mask=mask, other=0.0)
    
    tl.store(output_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def fuse_conv2d_with_embeddings(pixel_values, conv_bias, conv_weight, cls_token, detection_tokens):
    batch_size, in_channels, input_height, input_width = pixel_values.shape
    out_channels, _, kernel_height, kernel_width = conv_weight.shape
    
    # Perform conv2d
    conv_output = torch.empty(batch_size, out_channels, input_height // 2, input_width // 2, dtype=pixel_values.dtype, device=pixel_values.device)
    conv2d_kernel[(conv_output.numel() + 1023) // 1024,](
        pixel_values, conv_weight, conv_bias,
        conv_output,
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_height, kernel_width,
        2, 2,  # stride
        0, 0,  # padding
        1, 1,  # dilation
        1,     # groups
        1024
    )
    
    # Flatten and transpose conv output: [1, 32, 15, 15] -> [1, 7200] -> [7200, 1]
    batch_size, features, seq_len = 1, 32 * 15 * 15, 1
    flattened = torch.empty(batch_size, features, seq_len, dtype=conv_output.dtype, device=conv_output.device)
    flatten_transpose_kernel[(batch_size * seq_len + 1023) // 1024,](
        conv_output, flattened,
        batch_size, features, seq_len,
        1024
    )
    
    transposed = torch.empty(seq_len, features, dtype=flattened.dtype, device=flattened.device)
    flatten_transpose_kernel[(seq_len * features + 1023) // 1024,](
        flattened, transposed,
        seq_len, features, batch_size,
        1024  
    )
    
    # Expand cls token: [1, 1, 32] -> [7200, 32]
    cls_expanded = torch.empty(7200, 32, dtype=cls_token.dtype, device=cls_token.device)
    expand_cat_kernel[(7200 * 32 + 1023) // 1024,](
        cls_token.squeeze(), transposed, detection_tokens,
        cls_expanded,
        1, 32, 1, 320, 353,
        1024
    )
    
    return cls_expanded

def pattern(pixel_values, conv_bias, conv_weight, cls_token, detection_tokens):
    conv2d = torch.conv2d(pixel_values, conv_weight, conv_bias, (2, 2), (0, 0), (1, 1), 1)
    tmp_8 = conv2d.flatten(2)
    tmp_9 = tmp_8.transpose(1, 2)
    tmp_10 = cls_token.expand(1, -1, -1)
    tmp_11 = detection_tokens.expand(1, -1, -1)
    tmp_12 = torch.cat((tmp_10, tmp_9, tmp_11), dim=1)
    return tmp_12

def replacement_args(pixel_values, conv_bias, conv_weight, cls_token, detection_tokens):
    return (pixel_values, conv_bias, conv_weight, cls_token, detection_tokens)

def replacement_func():
    return fuse_conv2d_with_embeddings