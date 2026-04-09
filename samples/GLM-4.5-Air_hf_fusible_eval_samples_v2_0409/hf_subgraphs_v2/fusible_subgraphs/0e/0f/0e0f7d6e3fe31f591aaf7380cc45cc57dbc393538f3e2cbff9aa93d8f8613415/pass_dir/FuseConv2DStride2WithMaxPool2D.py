import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Match Conv2D stride=(2,2) + MaxPool2D pattern
    """
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (3, 3), (1, 1), 1)
    tmp_3 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_conv2d_maxpool_kernel(
    input_ptr,      # Input tensor [N, C_in, H_in, W_in]
    weight_ptr,     # Weight tensor [C_out, C_in, K_H, K_W]
    output_ptr,     # Output tensor [N, C_out, H_out, W_out]
    input_batch,    # Batch size N
    input_channels, # Input channels C_in 
    input_height,   # Input height H_in
    input_width,    # Input width W_in
    output_channels,# Output channels C_out
    weight_height,  # Kernel height K_H
    weight_width,   # Kernel width K_W
    output_height,  # Output height H_out
    output_width,   # Output width W_out
    BLOCK_SIZE_M: tl.constexpr,  # Number of programs (output channels)
    BLOCK_SIZE_N: tl.constexpr,  # Number of programs (batch size)
    BLOCK_SIZE_K: tl.constexpr,  # Reduction dimension
):
    # Program identifiers
    pid_m = tl.program_id(0)  # Output channel
    pid_n = tl.program_id(1)  # Batch
    
    # Range of output channels and batch elements this program handles
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    m_mask = m_offsets < output_channels
    n_mask = n_offsets < input_batch
    
    # Initialize output
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input channels (reduction dimension)
    for k in range(0, input_channels, BLOCK_SIZE_K):
        k_offsets = tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < input_channels
        
        # Load input tile
        input_tile = tl.load(
            input_ptr + n_offsets[:, None] * (input_channels * input_height * input_width) + k_offsets[None, :] * (input_height * input_width),
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Load weight tile
        weight_tile = tl.load(
            weight_ptr + m_offsets[:, None] * (input_channels * weight_height * weight_width) + k_offsets[None, :] * (weight_height * weight_width),
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Convolution operation: dot product over input channels
        acc += tl.dot(input_tile, weight_tile.T)
    
    # Apply ReLU (activation function commonly used after conv2d)
    acc = tl.maximum(acc, 0.0)
    
    # Store result
    output_base_offset = n_offsets[:, None] * (output_channels * output_height * output_width) + m_offsets[None, :] * (output_height * output_width)
    tl.store(
        output_ptr + output_base_offset,
        acc,
        mask=n_mask[:, None] & m_mask[None, :]
    )

@torch.fx.wrap
def fused_conv2d_maxpool_triton(in_0, in_1):
    # Get tensor shapes
    batch_size, in_channels, in_height, in_width = in_1.shape
    out_channels, _, weight_height, weight_width = in_0.shape
    
    # Calculate output dimensions for conv2d with stride=(2,2), padding=(3,3)
    conv_out_height = (in_height + 2*3 - weight_height) // 2 + 1
    conv_out_width = (in_width + 2*3 - weight_width) // 2 + 1
    
    # Calculate output dimensions for maxpool2d with kernel_size=3, stride=2, padding=1
    pooled_height = (conv_out_height + 2*1 - 3) // 2 + 1
    pooled_width = (conv_out_width + 2*1 - 3) // 2 + 1
    
    # Allocate output tensor
    output = torch.empty((batch_size, out_channels, pooled_height, pooled_width), 
                        dtype=in_0.dtype, device=in_0.device)
    
    # Set up grid and block sizes - smaller blocks for better GPU utilization
    BLOCK_SIZE_M = 32  # Number of output channels per program
    BLOCK_SIZE_N = 4   # Number of batch items per program  
    BLOCK_SIZE_K = 64  # Number of input channels for reduction
    
    # Calculate grid dimensions
    grid_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (batch_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_conv2d_maxpool_kernel[(grid_m, grid_n), (
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )](
        in_1, in_0, output,
        batch_size, in_channels, in_height, in_width,
        out_channels, weight_height, weight_width,
        pooled_height, pooled_width
    )
    
    return output

def replacement_func():
    return fused_conv2d_maxpool_triton