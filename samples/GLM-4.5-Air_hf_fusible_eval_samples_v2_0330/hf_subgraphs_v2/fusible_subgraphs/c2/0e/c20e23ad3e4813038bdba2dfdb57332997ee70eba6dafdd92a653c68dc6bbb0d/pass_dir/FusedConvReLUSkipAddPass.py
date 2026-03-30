import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor, skip_tensor):
    """
    Pattern to match: Conv2D -> ReLU -> Addition (skip connection)
    
    This matches:
    conv2d = torch.conv2d(in_3, in_1, in_0, (2, 2), (1, 1), (1, 1), 1)
    tmp_3 = torch.nn.functional.relu(conv2d, inplace = True)
    tmp_4 = in_2 + tmp_3
    
    Returns tmp_4 which is used in subsequent operations
    """
    conv2d = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (2, 2), (1, 1), (1, 1), 1)
    tmp_3 = torch.nn.functional.relu(conv2d, inplace = True)
    tmp_4 = skip_tensor + tmp_3
    return tmp_4

def replacement_args(input_tensor, weight_tensor, bias_tensor, skip_tensor):
    return (input_tensor, weight_tensor, bias_tensor, skip_tensor)

@triton.jit
def fused_conv_relu_skip_kernel(
    input_ptr,
    weight_ptr, 
    bias_ptr,
    skip_ptr,
    output_ptr,
    input_batch,
    input_channels,
    input_height,
    input_width,
    output_channels,
    kernel_height,
    kernel_width,
    output_height,
    output_width,
    stride_height,
    stride_width,
    padding_height,
    padding_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, output_height)
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, output_width)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    
    for k in range(0, input_channels, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, input_channels)
        
        # Load input block
        input_offsets = (
            pid_b * input_channels * input_height * input_width +
            (k + tl.arange(0, BLOCK_SIZE_K))[:, None, None] +
            tl.arange(0, kernel_height)[None, :, None] * input_channels * input_width +
            tl.arange(0, kernel_width)[None, None, :] * input_channels
        )
        input_block = tl.load(input_ptr + input_offsets.to(tl.int64), 
                           mask=(tl.arange(0, BLOCK_SIZE_K)[:, None, None] < (k_end - k)) &
                                (tl.arange(0, kernel_height)[None, :, None] < kernel_height) &
                                (tl.arange(0, kernel_width)[None, None, :] < kernel_width),
                           other=0.0, dtype=tl.float16)
        
        # Load weight block
        weight_offsets = (
            (k + tl.arange(0, BLOCK_SIZE_K))[:, None, None] * output_channels + 
            tl.arange(0, kernel_height)[None, :, None] * output_channels * output_width +
            tl.arange(0, kernel_width)[None, None, :] * output_channels
        )
        weight_block = tl.load(weight_ptr + weight_offsets.to(tl.int64),
                           mask=(tl.arange(0, BLOCK_SIZE_K)[:, None, None] < (k_end - k)) &
                                (tl.arange(0, kernel_height)[None, :, None] < kernel_height) &
                                (tl.arange(0, kernel_width)[None, None, :] < kernel_width),
                           other=0.0, dtype=tl.float16)
        
        # Load bias
        bias_offsets = n_start + tl.arange(0, min(BLOCK_SIZE_N, n_end - n_start))[None, :]
        bias_block = tl.load(bias_ptr + bias_offsets.to(tl.int64),
                           mask=tl.arange(0, min(BLOCK_SIZE_N, n_end - n_start))[:, None] < (n_end - n_start),
                           other=0.0, dtype=tl.float16)
        
        # Convolution + ReLU
        input_reshaped = input_block.to(tl.float32).reshape(BLOCK_SIZE_K, kernel_height * kernel_width)
        weight_reshaped = weight_block.to(tl.float32).reshape(BLOCK_SIZE_K, BLOCK_SIZE_N)
        conv_result = tl.dot(input_reshaped, weight_reshaped).to(tl.float16)
        relu_result = tl.maximum(conv_result, 0.0)
        biased_result = relu_result + bias_block
        
        accumulator += biased_result
    
    # Load skip connection
    skip_block = tl.zeros((m_end - m_start, n_end - n_start), dtype=tl.float16)
    if m_end <= input_height and n_end <= input_width:
        skip_offsets = (
            pid_b * input_channels * input_height * input_width +
            tl.arange(0, m_end - m_start)[:, None] * input_channels * input_width +
            tl.arange(0, n_end - n_start)[None, :] * input_channels
        )
        skip_block = tl.load(skip_ptr + skip_offsets.to(tl.int64),
                           mask=(tl.arange(0, m_end - m_start)[:, None] < (m_end - m_start)) &
                                (tl.arange(0, n_end - n_start)[None, :] < (n_end - n_start)),
                           other=0.0, dtype=tl.float16)
    
    # Add skip connection
    final_result = accumulator + skip_block
    
    # Store output
    output_offsets = (
        pid_b * output_channels * output_height * output_width +
        tl.arange(0, m_end - m_start)[:, None] * output_channels * output_width + 
        n_start + tl.arange(0, n_end - n_start)[None, :]
    )
    
    tl.store(output_ptr + output_offsets.to(tl.int64), final_result,
             mask=(tl.arange(0, m_end - m_start)[:, None] < (m_end - m_start)) &
                  (tl.arange(0, n_end - n_start)[None, :] < (n_end - n_start)))

@torch.fx.wrap
def fused_conv_relu_add(input_tensor, weight_tensor, bias_tensor, skip_tensor):
    input_batch, input_channels, input_height, input_width = input_tensor.shape
    output_channels, _, kernel_height, kernel_width = weight_tensor.shape
    output_height = input_height // stride_height
    output_width = input_width // stride_width
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 64
    
    num_programs_m = (output_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (output_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_programs_b = input_batch
    
    output = torch.empty((input_batch, output_channels, output_height, output_width), 
                        dtype=torch.float16, device=input_tensor.device)
    
    fused_conv_relu_skip_kernel[
        (num_programs_m, num_programs_n, num_programs_b)
    ](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        skip_ptr=skip_tensor,
        output_ptr=output,
        input_batch=input_batch,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        output_channels=output_channels,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        output_height=output_height,
        output_width=output_width,
        stride_height=2,
        stride_width=2,
        padding_height=1,
        padding_width=1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return fused_conv_relu_add