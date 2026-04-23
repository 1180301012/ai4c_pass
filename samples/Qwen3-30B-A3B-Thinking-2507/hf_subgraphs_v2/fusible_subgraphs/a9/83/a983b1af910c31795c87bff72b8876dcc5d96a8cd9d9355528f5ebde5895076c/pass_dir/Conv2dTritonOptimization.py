import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(input_ptr, weight_ptr, output_ptr,
                 batch_size, in_channels, out_channels, height, width,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    batch_offset = pid_batch * in_channels * height * width
    
    output_ptr += pid_batch * out_channels * height * width + m_start * height * width + n_start
    input_ptr += batch_offset + n_start
    weight_ptr += m_start * in_channels
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, in_channels, BLOCK_K):
        input_chunk = tl.load(
            input_ptr + k * height * width,
            mask=(k + tl.arange(0, BLOCK_K) < in_channels),
            other=0.0
        )
        weight_chunk = tl.load(
            weight_ptr + k,
            mask=(k + tl.arange(0, BLOCK_K) < in_channels),
            other=0.0
        )
        acc += input_chunk[None, :] * weight_chunk[:, None]
    
    tl.store(
        output_ptr,
        acc,
        mask=(m_start + tl.arange(0, BLOCK_M) < out_channels)[:, None] &
              (n_start + tl.arange(0, BLOCK_N) < height * width)[None, :]
    )

@torch.fx.wrap
def conv2d_triton(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups):
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight_tensor.shape[0]
    
    # Reshape weight to [out_channels, in_channels]
    weight_tensor = weight_tensor.squeeze()
    weight_tensor = weight_tensor.permute(1, 0).permute(1, 0)
    
    # Allocate output
    output_tensor = torch.empty(batch_size, out_channels, height, width, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Configure kernel grid
    grid = (
        triton.cdiv(out_channels, 16),
        triton.cdiv(height * width, 128),
        batch_size
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight_tensor,
        output_tensor,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        16, 128, 64
    )
    
    # Add bias
    if bias_tensor is not None:
        output_tensor += bias_tensor.view(1, out_channels, 1, 1)
    
    return output_tensor

def pattern(x, y, z):
    conv = torch.conv2d(x, y, z, (1, 1), (0, 0), (1, 1), 1)
    return conv

def replacement_args(x, y, z):
    return (x, y, z)

def replacement_func():
    return conv2d_triton