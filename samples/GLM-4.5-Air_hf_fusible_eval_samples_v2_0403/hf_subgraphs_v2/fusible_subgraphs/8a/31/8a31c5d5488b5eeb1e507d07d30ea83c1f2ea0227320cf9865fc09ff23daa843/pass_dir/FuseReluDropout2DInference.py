import torch
import triton
import triton.language as tl
import math

# Pattern matching function - must exactly match the model.py computation
def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized fused kernel for ReLU + Dropout2D (inference mode)
@triton.jit
def fused_relu_dropout_kernel(
    input_ptr,
    relu_output_ptr,
    dropout_output_ptr,
    batch_size,
    channels,
    height,
    width,
    dropout_scale: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # grid setup
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # compute ranges
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, batch_size)
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, channels)
    
    # pointer initialization (2D spatial layout for each batch-channel pair)
    input_base = input_ptr + (m_start * channels + n_start) * height * width
    
    relu_output_base = relu_output_ptr + (m_start * channels + n_start) * height * width
    dropout_output_base = dropout_output_ptr + (m_start * channels + n_start) * height * width
    
    # calculate output size for this block
    block_height = height
    block_width = width
    
    # offset computation
    offsets_h = tl.arange(0, block_height)
    offsets_w = tl.arange(0, block_width)
    offsets_hw = offsets_h[:, None] * width + offsets_w[None, :]
    
    # compute effective range
    current_batch_channels = (m_end - m_start) * (n_end - n_start)
    
    # process each batch-channel pair in the block
    for bc_idx in range(current_batch_channels):
        # pointer for current batch-channel
        input_ptr_bc = input_base + bc_idx * height * width
        relu_ptr_bc = relu_output_base + bc_idx * height * width
        dropout_ptr_bc = dropout_output_base + bc_idx * height * width
        
        # load input
        x = tl.load(input_ptr_bc + offsets_hw, mask=(offsets_hw < height * width), other=0.0)
        
        # fused relu and scaling
        relu_out = tl.maximum(x, 0.0)
        dropout_out = relu_out * dropout_scale
        
        # store both outputs
        tl.store(relu_ptr_bc + offsets_hw, relu_out, mask=(offsets_hw < height * width))
        tl.store(dropout_ptr_bc + offsets_hw, dropout_out, mask=(offsets_hw < height * width))

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_relu_dropout_wrapper(in_0):
    batch_size, channels, height, width = in_0.shape
    dropout_scale = 0.9  # 1 - 0.1 probability
    
    # create output tensors
    relu_out = torch.empty_like(in_0, device=in_0.device)
    dropout_out = torch.empty_like(in_0, device=in_0.device)
    
    # optimal block sizes for 512 channels, 64x64 spatial
    BLOCK_SIZE_M = 32  # batch size block
    BLOCK_SIZE_N = 32  # channels block
    
    # calculate grid dimensions
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # launch kernel
    fused_relu_dropout_kernel[(grid_m, grid_n)](
        in_0,
        relu_out,
        dropout_out,
        batch_size,
        channels,
        height,
        width,
        dropout_scale,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    
    return (dropout_out, relu_out)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_relu_dropout_wrapper