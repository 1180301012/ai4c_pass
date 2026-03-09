import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern matching for conv2d -> gelu -> (no-op dropout) sequence"""
    # Handle depthwise conv case based on weight tensor shape
    if in_1.shape[0] == in_2.shape[1] and in_1.shape[1] == 1:
        groups = in_1.shape[0]  # depthwise convolution
    else:
        groups = in_1.shape[0]  # regular convolution
    
    # conv2d operation
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), groups)
    # gelu activation
    tmp_3 = torch.nn.functional.gelu(tmp_2)
    # no-op dropout (p=0.0) - this is identity operation
    tmp_4 = tmp_3  # dropout with p=0.0 is identity
    return tmp_4,

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the replacement kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv2d_gelu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused conv2d + gelu kernel with optimized Triton implementation"""
    
    # Calculate program indices
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Compute memory addresses for output tile
    output_base = m * BLOCK_SIZE_M * width + n * BLOCK_SIZE_N
    output_ptrs = output_base + tl.arange(0, BLOCK_SIZE_M)[:, None] * width + tl.arange(0, BLOCK_SIZE_N)[None, :]
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Load weight tile
    weight_base = n * BLOCK_SIZE_N * kernel_size * kernel_size
    weight_ptrs = weight_base + tl.arange(0, BLOCK_SIZE_K)[:, None] * kernel_size * kernel_size + tl.arange(0, kernel_size * kernel_size)[None, :]
    weight = tl.load(weight_ptrs, mask=tl.arange(0, BLOCK_SIZE_K)[:, None] < in_channels * kernel_size * kernel_size, other=0.0)
    
    # Load bias (if available)
    bias_ptr_n = bias_ptr + n * BLOCK_SIZE_N
    bias = tl.load(bias_ptr_n + tl.arange(0, BLOCK_SIZE_N), mask=tl.arange(0, BLOCK_SIZE_N) < out_channels, other=0.0)
    
    # Main convolution loop over input channels
    for k in range(0, in_channels, BLOCK_SIZE_K):
        # Load input tile
        input_base = m * BLOCK_SIZE_M * width + k * width * height
        input_ptrs = (
            input_base + tl.arange(0, BLOCK_SIZE_M)[:, None] * width * height + 
            tl.arange(0, height)[None, :] * width + 
            tl.arange(0, width)[None, :]
        )
        input_tile = tl.load(input_ptrs, mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < height * width, other=0.0)
        
        # Convolution operation: input @ weight
        for i in range(BLOCK_SIZE_M):
            for j in range(BLOCK_SIZE_N):
                for kk in range(BLOCK_SIZE_K):
                    if i < BLOCK_SIZE_M and j < BLOCK_SIZE_N and kk < BLOCK_SIZE_K:
                        accumulator[i, j] += input_tile[i] * weight[kk, j]
    
    # Add bias
    accumulator += bias[None, :]
    
    # Apply GELU activation
    gelu_out = accumulator * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (accumulator + 0.044715 * tl.power(accumulator, 3))))
    
    # Store output
    tl.store(output_ptr + output_ptrs, gelu_out, mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < height * width)

@torch.fx.wrap
def fused_conv2d_gelu(in_0, in_1, in_2):
    """Fused conv2d + gelu implementation"""
    # Get input shapes
    batch_size, in_channels, height, width = in_2.shape
    out_channels, _, kernel_size_h, kernel_size_w = in_1.shape
    
    # Handle depthwise conv case (groups == in_channels)
    if in_1.shape[0] == in_channels and in_1.shape[1] == 1:
        groups = in_channels
    else:
        groups = in_1.shape[0]
    
    # For simplicity, assume same kernel size, stride, padding
    kernel_size = kernel_size_h
    stride_h, stride_w = 1, 1
    padding_h, padding_w = 1, 1
    
    # Calculate output dimensions with padding
    out_height = (height + 2 * padding_h - kernel_size_h) // stride_h + 1
    out_width = (width + 2 * padding_w - kernel_size_w) // stride_w + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), dtype=in_2.dtype, device=in_2.device)
    
    # Choose block sizes for Triton kernel
    BLOCK_SIZE_M = 64  # batch dimension tile
    BLOCK_SIZE_N = 64  # output channels tile  
    BLOCK_SIZE_K = 32  # input channels tile
    
    # Launch kernel
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    fused_conv2d_gelu_kernel[(grid_m, grid_n)](
        in_2,
        in_1, 
        in_0,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride_h,
        padding_h,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return fused_conv2d_gelu