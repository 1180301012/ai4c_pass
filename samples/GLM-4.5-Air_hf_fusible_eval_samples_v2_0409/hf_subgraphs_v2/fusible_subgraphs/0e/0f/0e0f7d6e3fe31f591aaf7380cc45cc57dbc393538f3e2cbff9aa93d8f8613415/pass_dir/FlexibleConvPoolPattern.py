import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Match Conv2D stride=(1,1) + MaxPool2D pattern
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_conv2d_maxpool_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_batch,
    input_channels,
    input_height,
    input_width,
    output_channels,
    weight_height,
    weight_width,
    output_height,
    output_width,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offsets < output_channels
    n_mask = n_offsets < input_batch
    
    # Initialize output
    output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Convolution operation
    for k in range(0, input_channels, BLOCK_SIZE_K):
        k_offsets = tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < input_channels
        
        # Load input data
        input_data = tl.load(
            input_ptr + (n_offsets[:, None] * input_channels + k_offsets[None, :]) * (input_height * input_width),
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Load weights
        weights = tl.load(
            weight_ptr + (m_offsets[:, None] * input_channels + k_offsets[None, :]) * (weight_height * weight_width),
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Compute convolution
        output += tl.dot(input_data, weights.T)
    
    # Apply ReLU activation
    output = tl.maximum(output, 0.0)
    
    # Store output
    base_offset = (n_offsets[:, None] * output_channels + m_offsets[None, :]) * (output_height * output_width)
    tl.store(
        output_ptr + base_offset,
        output,
        mask=n_mask[:, None] & m_mask[None, :]
    )

@torch.fx.wrap
def fused_conv2d_maxpool_triton(in_0, in_1):
    batch_size, in_channels, in_height, in_width = in_1.shape
    out_channels, _, weight_height, weight_width = in_0.shape
    
    # For stride=1, padding=1
    conv_out_height = (in_height + 2*1 - weight_height) // 1 + 1
    conv_out_width = (in_width + 2*1 - weight_width) // 1 + 1
    
    # Max pooling with kernel_size=3, stride=2, padding=1
    pooled_height = (conv_out_height + 2*1 - 3) // 2 + 1
    pooled_width = (conv_out_width + 2*1 - 3) // 2 + 1
    
    output = torch.empty((batch_size, out_channels, pooled_height, pooled_width),
                        dtype=in_0.dtype, device=in_0.device)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 8
    BLOCK_SIZE_K = 32
    
    grid_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (batch_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    fused_conv2d_maxpool_kernel[(grid_m, grid_n), (
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )](
        in_1, in_0, output,
        batch_size, in_channels, in_height, in_width,
        out_channels, weight_height, weight_width,
        pooled_height, pooled_width,
        1, 1,  # stride_h, stride_w
        1, 1   # pad_h, pad_w
    )
    
    return output

def replacement_func():
    return fused_conv2d_maxpool_triton