import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, stride=1, padding=64, dilation=1, groups=16):
    """Pattern to match conv1d followed by slice operation"""
    conv = torch.conv1d(x, weight, bias, stride, padding, dilation, groups)
    sliced = conv[:, :, :-1]
    return sliced

def replacement_args(x, weight, bias, stride=1, padding=64, dilation=1, groups=16):
    return (x, weight, bias, stride, padding, dilation, groups)

@triton.jit
def conv1d_slice_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels,
    seq_len,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized conv1d with integrated slicing to avoid last element"""
    # Calculate the effective output length without the last element
    effective_seq_len = seq_len - 1
    
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Range of columns each program should process
    cols_per_program = (effective_seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    block_start_n = pid_n * cols_per_program
    offsets_n = block_start_n * stride + tl.arange(0, BLOCK_SIZE_N) * stride
    
    # Each program works on one batch and one output channel group
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N
    
    k_offset = tl.arange(0, BLOCK_SIZE_K)
    
    # Load input x [batch, in_channels, seq_len]
    x_ptrs = x_ptr + m_offset[:, None] * in_channels * seq_len + \
             k_offset[None, :] * seq_len + offsets_n[:, None] + dilation * tl.arange(0, BLOCK_SIZE_K)[None, :]
    x = tl.load(x_ptrs, mask=(m_offset[:, None] < batch_size)[:, :, None] & 
                          (k_offset[None, :] < in_channels)[:, :, None] & 
                          (offsets_n[:, None] < effective_seq_len)[:, :, None], 
                other=0.0)
    
    # Load weights [out_channels, kernel_size, in_channels/groups]
    weight_ptrs = weight_ptr + pid_n * BLOCK_SIZE_K + k_offset[:, None]
    weight = tl.load(weight_ptrs, mask=k_offset[:, None] < (kernel_size * in_channels // groups), 
                     other=0.0)
    
    # Load bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), 
                          mask=(pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) < out_channels,
                          other=0.0)
    else:
        bias_val = 0.0
    
    # Reshape for matrix multiplication
    x = x.reshape((BLOCK_SIZE_M, BLOCK_SIZE_K))
    weight = weight.reshape((BLOCK_SIZE_K, BLOCK_SIZE_N))
    
    # Compute convolution
    output = tl.dot(x, weight)
    output += bias_val[:, None]
    
    # Store output [batch, out_channels, effective_seq_len]
    out_ptrs = out_ptr + m_offset[:, None] * out_channels * effective_seq_len + \
              tl.arange(0, BLOCK_SIZE_N)[None, :] + \
              n_offset[None, :] * out_channels
    tl.store(out_ptrs, output, mask=(m_offset[:, None] < batch_size) & 
                                   (tl.arange(0, BLOCK_SIZE_N)[None, :] < effective_seq_len))

@torch.fx.wrap
def triton_conv1d_slice(x, weight, bias, stride=1, padding=64, dilation=1, groups=16):
    """Wrapper function for optimized conv1d slice"""
    batch_size, in_channels, seq_len = x.shape
    out_channels, kernel_size, _ = weight.shape
    
    # Output sequence length without the last element
    effective_seq_len = seq_len - 1
    
    # Adjust padding to account for slicing
    adjusted_padding = padding - 1 if dilation == 1 else padding
    
    out = torch.empty((batch_size, out_channels, effective_seq_len), 
                     dtype=x.dtype, device=x.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    grid = (triton.cdiv(batch_size * out_channels, BLOCK_SIZE_M),
            triton.cdiv(effective_seq_len, BLOCK_SIZE_N))
    
    conv1d_slice_kernel[grid](
        x,
        weight,
        bias,
        out,
        batch_size,
        in_channels,
        seq_len,
        out_channels,
        kernel_size,
        stride,
        adjusted_padding,
        dilation,
        groups,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return triton_conv1d_slice