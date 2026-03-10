import torch
import triton
import triton.language as tl
from torch import fx

def pattern(x, weight, bias, stride, padding, dilation, groups):
    # This pattern matches the Conv2D + Flatten(2) + Transpose(1,2) sequence
    conv_out = torch.conv2d(x, weight, bias, stride, padding, dilation, groups)
    flatten_out = conv_out.flatten(2)
    transpose_out = flatten_out.transpose(1, 2)
    return conv_out.transpose(1, 2), transpose_out

def replacement_args(x, weight, bias, stride, padding, dilation, groups):
    return (x, weight, bias, stride, padding, dilation, groups)

@triton.jit
def fused_conv_flatten_transpose_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels, in_height, in_width,
    out_channels, kernel_height, kernel_width,
    stride_height, stride_width,
    padding_height, padding_width,
    dilation_height, dilation_width,
    groups,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Grid setup for batch and output positions
    m = tl.program_id(0)  # Batch dimension
    n = tl.program_id(1)  # Output position (flattened spatial)
    k = tl.program_id(2)  # Output channel
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
    out_width = (in_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1
    spatial_size = out_height * out_width
    
    # Compute memory offsets
    x_offset = m * in_channels * in_height * in_width
    weight_offset = k * in_channels * kernel_height * kernel_width // groups
    out_offset = m * spatial_size
    result_offset = out_offset + n * spatial_size + k
    
    # Create masks
    n_mask = n < spatial_size
    k_mask = k < out_channels
    
    if not (n_mask and k_mask):
        return
    
    # Load kernel weights
    weight_vals = tl.load(weight_ptr + weight_offset + tl.arange(0, kernel_height * kernel_width * in_channels // groups)[:, None],
                         mask=tl.arange(0, kernel_height * kernel_width * in_channels // groups)[:, None] < (kernel_height * kernel_width * in_channels // groups),
                         other=0.0)
    
    # Reshape weight for conv operation
    weight_reshaped = weight_vals.reshape(in_channels // groups, kernel_height, kernel_width, out_channels // groups)
    
    # Initialize accumulator
    acc = 0.0
    
    # Convolution computation (simplified for this specific pattern)
    for c_in in range(in_channels // groups):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input coordinates
                ih = n // out_width * stride_height + kh * dilation_height - padding_height
                iw = n % out_width * stride_width + kw * dilation_width - padding_width
                
                if 0 <= ih < in_height and 0 <= iw < in_width:
                    x_idx = x_offset + c_in * in_height * in_width + ih * in_width + iw
                    x_val = tl.load(x_ptr + x_idx, other=0.0)
                    weight_idx = weight_offset + c_in * kernel_height * kernel_width + kh * kernel_width + kw
                    weight_val = tl.load(weight_ptr + weight_idx, other=0.0)
                    acc += x_val * weight_val
    
    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + k, other=0.0)
        acc += bias_val
    
    # Store result
    tl.store(out_ptr + result_offset, acc)

@torch.fx.wrap  
def optimized_fused_conv_flatten(x, weight, bias, stride=(4,4), padding=(0,0), dilation=(1,1), groups=1):
    # Get input dimensions
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, kernel_channels, kernel_height, kernel_width = weight.shape
    
    # Verify compatibility
    assert in_channels == kernel_channels * groups, "Input channels must match weight channels * groups"
    assert stride == (4,4) and padding == (0,0) and dilation == (1,1) and groups == 1, "This kernel is optimized for specific parameters"
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    spatial_size = out_height * out_width
    
    # Output tensors
    # conv_output: [batch_size, out_channels, spatial_size] (already transposed)
    # flatten_transpose_output: [batch_size, spatial_size, out_channels] 
    conv_output = torch.empty(batch_size, out_channels, spatial_size, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
    flatten_transpose_output = torch.empty(batch_size, spatial_size, out_channels, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
    
    # Triton kernel launch parameters
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Grid setup
    grid_m = batch_size
    grid_n = spatial_size
    grid_k = out_channels
    
    fused_conv_flatten_transpose_kernel[(grid_m, grid_n, grid_k)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=flatten_transpose_output,  # Directly compute the transposed flattened output
        batch_size=batch_size,
        in_channels=in_channels, in_height=in_height, in_width=in_width,
        out_channels=out_channels, kernel_height=kernel_height, kernel_width=kernel_width,
        stride_height=stride[0], stride_width=stride[1],
        padding_height=padding[0], padding_width=padding[1],
        dilation_height=dilation[0], dilation_width=dilation[1],
        groups=groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Compute conv output for return (transpose of flattened spatial)
    conv_output = flatten_transpose_output.transpose(1, 2)
    
    return conv_output, flatten_transpose_output

def replacement_func():
    return optimized_fused_conv_flatten