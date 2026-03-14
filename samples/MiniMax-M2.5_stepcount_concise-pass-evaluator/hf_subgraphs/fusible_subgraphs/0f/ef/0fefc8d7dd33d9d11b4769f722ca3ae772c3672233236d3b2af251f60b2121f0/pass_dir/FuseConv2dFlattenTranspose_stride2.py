import torch
import triton
import triton.language as tl


# Optimized Triton kernel for fused flatten + transpose
# This fuses the flatten(2) and transpose operations which are memory-bound
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=8),
    ],
    key=['seq_len'],
)
@triton.jit
def fused_flatten_transpose_kernel(
    input_ptr, output_ptr,
    batch_size, channels, height, width,
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    output_batch_stride, output_channel_stride, output_seq_stride,
    seq_len: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    Fused flatten + transpose kernel.
    Input: (batch, channels, height, width)
    Output: (batch, channels, height*width) - flatten(2) then already in correct order
            The model.py does: flatten(2) -> transpose(1, 2)
            flatten(2): (B, C, H, W) -> (B, H*W, C)
            transpose(1, 2): (B, H*W, C) -> (B, C, H*W)
            
    So our output should be: (B, C, H*W)
    """
    # Calculate global position
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate base offset for this block
    base_seq = seq_idx * BLOCK_SIZE
    
    # Process BLOCK_SIZE elements
    for i in range(BLOCK_SIZE):
        seq_offset = base_seq + i
        if seq_offset < seq_len:
            # Convert flat seq index to (h, w)
            h = seq_offset // width
            w = seq_offset % width
            
            # Load from input: (batch, channels, height, width)
            for c in range(channels):
                in_ptr = batch_idx * input_batch_stride + c * input_channel_stride + h * input_h_stride + w * input_w_stride
                val = tl.load(input_ptr + in_ptr)
                
                # Store to output: (batch, channels, seq)
                out_ptr = batch_idx * output_batch_stride + c * output_channel_stride + seq_offset * output_seq_stride
                tl.store(output_ptr + out_ptr, val)


def fuse_conv_flatten_transpose(input_tensor, weight, bias, stride, padding, dilation, groups):
    """
    Fused Conv2D + Flatten + Transpose operation.
    
    The computation:
    1. Conv2D: (B, C_in, H, W) -> (B, C_out, H', W')
    2. Flatten: (B, C_out, H', W') -> (B, H'*W', C_out)
    3. Transpose: (B, H'*W', C_out) -> (B, C_out, H'*W')
    
    Returns (conv_out, flattened, transposed) to match pattern outputs.
    """
    # Step 1: Use PyTorch's optimized conv2d (cuDNN backend)
    conv_out = torch.nn.functional.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)
    
    # Step 2 & 3: Fused flatten + transpose using Triton kernel
    batch_size, out_channels, out_h, out_w = conv_out.shape
    seq_len = out_h * out_w
    
    # Allocate output: (batch, channels, seq)
    transposed = torch.empty((batch_size, out_channels, seq_len), 
                             device=conv_out.device, dtype=conv_out.dtype)
    
    # Get strides
    input_batch_stride = conv_out.stride(0)
    input_channel_stride = conv_out.stride(1)
    input_h_stride = conv_out.stride(2)
    input_w_stride = conv_out.stride(3)
    
    output_batch_stride = transposed.stride(0)
    output_channel_stride = transposed.stride(1)
    output_seq_stride = transposed.stride(2)
    
    # Choose block size based on sequence length
    if seq_len <= 256:
        BLOCK_SIZE = 256
    elif seq_len <= 512:
        BLOCK_SIZE = 512
    elif seq_len <= 1024:
        BLOCK_SIZE = 1024
    elif seq_len <= 2048:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096
    
    # Calculate grid
    grid = (batch_size, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    fused_flatten_transpose_kernel[grid](
        conv_out, transposed,
        batch_size, out_channels, out_h, out_w,
        input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
        output_batch_stride, output_channel_stride, output_seq_stride,
        seq_len, BLOCK_SIZE,
    )
    
    # Compute flattened intermediate (B, seq, C) for compatibility
    # This is tmp_7 in the original pattern
    flattened = conv_out.view(batch_size, out_channels, -1).transpose(1, 2)
    
    return conv_out, flattened, transposed


# Pattern matching function - matches conv2d -> flatten(2) -> transpose(1, 2) with stride (2,2)
def pattern(in_5, in_3, in_2):
    """
    Match the pattern with stride (2, 2): conv2d -> flatten(2) -> transpose(1, 2)
    
    This pattern is used in intermediate patch embedding layers.
    The model uses:
        tmp_5 = torch.conv2d(in_5, tmp_3, tmp_2, (2, 2), (0, 0), (1, 1), 1)
        tmp_6 = tmp_5.flatten(2)
        tmp_7 = tmp_6.transpose(1, 2)
    
    We fuse flatten + transpose into a single optimized kernel.
    Note: The conv2d is kept as-is (using PyTorch's optimized cuDNN implementation).
    """
    tmp_6 = torch.conv2d(in_5, in_3, in_2, (2, 2), (0, 0), (1, 1), 1)
    tmp_7 = tmp_6.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    return tmp_6, tmp_7, tmp_8


def replacement_args(in_5, in_3, in_2):
    """
    Extract arguments needed for the replacement function.
    """
    return (in_5, in_3, in_2)


def replacement_func():
    """
    Returns the optimized replacement function.
    """
    return fuse_conv_flatten_transpose