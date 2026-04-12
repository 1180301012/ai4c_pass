import torch
import triton
import triton.language as tl

def pattern(in_4, in_3, in_2):
    """Match Conv2D + residual addition pattern"""
    conv2d = torch.conv2d(in_4, in_3, in_2, (1, 1), (1, 1), (1, 1), 768)
    tmp_5 = conv2d + in_4
    return tmp_5

def replacement_args(in_4, in_3, in_2):
    return (in_4, in_3, in_2)

@triton.jit
def conv2d_residual_kernel(
    x_ptr,          # input tensor [N, C, H, W]
    weight_ptr,     # weight tensor [C_out, 1, K, K]
    bias_ptr,       # bias tensor [C_out]
    out_ptr,        # output tensor [N, C_out, H, W]
    n_channels_in,  # input channels
    n_channels_out, # output channels  
    height,         # input height
    width,          # input width
    eps,            # for bfloat16 conversion
    BLOCK_SIZE: tl.constexpr,
):
    """1x1 Conv2D with residual connection"""
    # Program identifiers for 2D grid
    b = tl.program_id(0)  # batch
    c = tl.program_id(1)  # output channel
    h0 = tl.program_id(2)  # start row for this block
    w0 = tl.program_id(3)  # start col for this block
    
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    
    # Coordinate offsets within this block
    h = h0 * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w = w0 * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    h_mask = h < height
    w_mask = w < width
    
    # Flatten coordinates for efficient memory access
    coords_x = b * height * width * n_channels_out + \
               c * height * width + \
               h[:, None] * width + w[None, :]
    coords_y = b * height * width * n_channels_in + \
               c * height * width + \
               h[:, None] * width + w[None, :]
    
    # Load input data
    x_data = tl.load(
        x_ptr + coords_y, 
        mask=h_mask[:, None] & w_mask[None, :], 
        other=0.0,
        eviction_policy='evict_last'
    )
    
    # Load weight (simplified for 1x1 kernel)
    weight_data = tl.load(weight_ptr + c, other=0.0)
    bias_data = tl.load(bias_ptr + c, other=0.0)
    
    # Apply 1x1 convolution (element-wise multiplication + sum)
    conv_result = 0.0
    if x_data.dtype == torch.bfloat16:
        conv_result = tl.sum(x_data.astype(tl.float32) * weight_data.astype(tl.float32), axis=(0, 1))
    else:
        conv_result = tl.sum(x_data * weight_data, axis=(0, 1))
    
    # Add bias
    conv_result += bias_data
    
    # Add residual connection (ensure same dtype)
    residual = x_data
    if residual.dtype != conv_result.dtype:
        residual = residual.astype(conv_result.dtype)
    
    result = conv_result + residual.reshape(conv_result.shape)
    
    # Store result
    tl.store(out_ptr + coords_x, result, mask=h_mask[:, None] & w_mask[None, :])

@torch.fx.wrap
def fused_conv2d_residual(x, weight, bias):
    """Fused 1x1 Conv2D with residual connection using Triton"""
    input_shape = x.shape
    if len(x.shape) != 4:
        raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")
    
    N, C_in, H, W = input_shape
    C_out = weight.shape[0]  # First dimension is output channels
    
    # Set block sizes for GPU efficiency
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    grid_x = (H + 15) // 16
    grid_y = (W + 15) // 16
    grid_z = (N * C_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty((N, C_out, H, W), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    eps = 1e-5 if x.dtype == torch.bfloat16 else 0.0
    
    conv2d_residual_kernel[(grid_z, 1, grid_x, grid_y)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_channels_in=C_in,
        n_channels_out=C_out,
        height=H,
        width=W,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_conv2d_residual