import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    """
    Pattern to match Conv2D operation
    This matches the expensive conv2d operation in the original model
    """
    conv2d_result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (2, 2), (1, 1), (1, 1), 1)
    return conv2d_result

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def optimized_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_batch,
    input_channels,
    input_height,
    input_width,
    output_channels,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    output_height,
    output_width,
    stride_height,
    stride_width,
    padding_height,
    padding_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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
        
        # Calculate input region with padding
        input_start_h = m_start * stride_height - padding_height
        input_start_w = n_start * stride_width - padding_width
        input_end_h = min(input_start_h + kernel_height, input_height)
        input_end_w = min(input_start_w + kernel_width, input_width)
        
        # Load input block directly with boundary handling
        input_block = tl.zeros((k_end - k, kernel_height, kernel_width), dtype=tl.float16)
        
        if input_start_h >= 0 and input_start_w >= 0 and input_end_h <= input_height and input_end_w <= input_width:
            # Normal case: no boundary issues
            input_offsets = (
                pid_b * input_channels * input_height * input_width +
                (k + tl.arange(0, k_end - k))[:, None, None] +
                (input_start_h + tl.arange(0, kernel_height))[None, :, None] * input_channels * input_width +
                (input_start_w + tl.arange(0, kernel_width))[None, None, :] * input_channels
            )
            input_block = tl.load(
                input_ptr + input_offsets.to(tl.int64),
                mask=(tl.arange(0, k_end - k)[:, None, None] < (k_end - k)) &
                     (tl.arange(0, kernel_height)[None, :, None] < kernel_height) &
                     (tl.arange(0, kernel_width)[None, None, :] < kernel_width),
                other=0.0, dtype=tl.float16
            )
        
        # Load weight block
        weight_offsets = (
            (k + tl.arange(0, BLOCK_SIZE_K))[:, None, None] * output_channels + 
            tl.arange(0, kernel_height)[None, :, None] * output_channels * output_width +
            tl.arange(0, kernel_width)[None, None, :] * output_channels
        )
        weight_block = tl.load(
            weight_ptr + weight_offsets.to(tl.int64),
            mask=(tl.arange(0, BLOCK_SIZE_K)[:, None, None] < (k_end - k)) &
                 (tl.arange(0, kernel_height)[None, :, None] < kernel_height) &
                 (tl.arange(0, kernel_width)[None, None, :] < kernel_width),
            other=0.0, dtype=tl.float16
        )
        
        # Perform matrix multiplication
        input_reshaped = input_block.to(tl.float32).reshape(k_end - k, kernel_height * kernel_width)
        weight_reshaped = weight_block.to(tl.float32).reshape(k_end - k, BLOCK_SIZE_N)
        conv_result = tl.dot(input_reshaped, weight_reshaped).to(tl.float16)
        
        # Load bias for this output slice
        bias_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
        if n_end - n_start <= BLOCK_SIZE_N:
            bias_offsets = n_start + tl.arange(0, n_end - n_start)[None, :]
            bias_block = tl.load(
                bias_ptr + bias_offsets.to(tl.int64),
                mask=tl.arange(0, n_end - n_start)[None, :] < (n_end - n_start),
                other=0.0, dtype=tl.float16
            )
        
        # Add bias and accumulate
        accumulator += conv_result + bias_block
    
    # Store output
    output_offsets = (
        pid_b * output_channels * output_height * output_width +
        tl.arange(0, m_end - m_start)[:, None] * output_channels * output_width + 
        n_start + tl.arange(0, n_end - n_start)[None, :]
    )
    
    tl.store(
        output_ptr + output_offsets.to(tl.int64), 
        accumulator, 
        mask=(tl.arange(0, m_end - m_start)[:, None] < (m_end - m_start)) &
             (tl.arange(0, n_end - n_start)[None, :] < (n_end - n_start))
    )

@torch.fx.wrap
def triton_conv2d(input_tensor, weight_tensor, bias_tensor):
    input_batch, input_channels, input_height, input_width = input_tensor.shape
    output_channels, _, kernel_height, kernel_width = weight_tensor.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * 1 - kernel_height) // 2 + 1  # With padding and stride
    output_width = (input_width + 2 * 1 - kernel_width) // 2 + 1
    
    # Optimized block sizes for Conv2D
    BLOCK_SIZE_M = 16   # Output height dimension
    BLOCK_SIZE_N = 16   # Output width dimension
    BLOCK_SIZE_K = 64   # Input channels (reduction dimension)
    
    num_programs_m = (output_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (output_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_programs_b = input_batch
    
    output = torch.empty((input_batch, output_channels, output_height, output_width), 
                        dtype=torch.float16, device=input_tensor.device)
    
    optimized_conv2d_kernel[
        (num_programs_m, num_programs_n, num_programs_b)
    ](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
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
    return triton_conv2d