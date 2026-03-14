import torch
import triton
import triton.language as tl

def pattern(input_tensor, conv_weight, arg2, arg3, arg4, arg5, arg6, arg7):
    # Pattern matching: Conv2D -> Flatten(2) -> Transpose(1,2)
    # This represents the patch embedding computation in ViT
    # The stride will be determined by the actual kernel size used in the model
    conv_output = torch.conv2d(input_tensor, conv_weight, None, (16, 16), (0, 0), (1, 1), 1)
    flattened = conv_output.flatten(2)
    transposed = flattened.transpose(1, 2)
    return transposed

def replacement_args(input_tensor, conv_weight, arg2, arg3, arg4, arg5, arg6, arg7):
    return (input_tensor, conv_weight)

@triton.jit
def fused_conv_flatten_transpose_kernel(
    input_ptr,           # [N, C_in, H_in, W_in]
    weight_ptr,          # [C_out, C_in, KH, KW] 
    output_ptr,          # [N, L, C_out] where L = H_out * W_out
    n_batch,             # Batch size (N)
    n_in_channels,       # Input channels (C_in)
    in_height,           # Input height (H_in)
    in_width,            # Input width (W_in)
    n_out_channels,      # Output channels (C_out)
    kernel_height,       # Kernel height (KH)
    kernel_width,        # Kernel width (KW)
    stride_height,       # Stride height
    stride_width,        # Stride width
    BLOCK_SIZE_M: tl.constexpr,      # Block size for output sequence dimension
    BLOCK_SIZE_N: tl.constexpr,      # Block size for embedding dimension
):
    # Each program handles a block of output sequence elements
    pid_m = tl.program_id(0)  # Output sequence position
    pid_n = tl.program_id(1)  # Output channel dimension
    
    # Calculate output spatial dimensions based on stride
    out_height = (in_height - kernel_height) // stride_height + 1
    out_width = (in_width - kernel_width) // stride_width + 1
    output_seq_len = out_height * out_width
    
    # Initialize output ptr for this program
    output_base = output_ptr + pid_m * n_out_channels * n_batch + pid_n
    
    # Process input tensor tile by output sequence tile
    for m_base in tl.range(0, output_seq_len, BLOCK_SIZE_M):
        m_offsets = m_base + tl.arange(0, BLOCK_SIZE_M)
        if m_offsets < output_seq_len:
            # Convert 1D sequence offset to 2D spatial location
            h = m_offsets // out_width
            w = m_offsets % out_width
            
            # Calculate input spatial locations (considering stride)
            in_h = h * stride_height
            in_w = w * stride_width
            
            # Initialize input pointer for this spatial location
            input_base = input_ptr + (in_h * in_width + in_w) * n_in_channels
            
            for n_base in tl.range(0, n_out_channels, BLOCK_SIZE_N):
                n_offsets = n_base + tl.arange(0, BLOCK_SIZE_N)
                if n_offsets < n_out_channels:
                    # Load weight tile: [out_channels, in_channels, KH, KW] 
                    # We need weight[pid_n, :, :, :] for all input channels
                    weight_base = weight_ptr + pid_n * n_in_channels * kernel_height * kernel_width
                    weight_offsets = (n_offsets[:, None, None, None] * n_in_channels * kernel_height * kernel_width +
                                     tl.arange(0, n_in_channels)[None, :, None, None] * kernel_height * kernel_width +
                                     tl.arange(0, kernel_height)[None, None, :, None] * kernel_width +
                                     tl.arange(0, kernel_width)[None, None, None, :])
                    
                    weights = tl.load(weight_base + weight_offsets, mask=(n_offsets[:, None, None, None] < n_out_channels)[:, 0, 0, 0], other=0.0)
                    
                    # Compute convolution output for this spatial location and output channel
                    # We need to sum over input channels and kernel spatial dimensions
                    acc = tl.zeros([n_offsets.shape[0]], dtype=tl.float32)
                    
                    # Iterate over input channels
                    for c_in in range(n_in_channels):
                        # Load input patch: [KH, KW]
                        input_patch_ptr = input_base + c_in
                        input_patch = tl.load(input_patch_ptr + tl.arange(0, kernel_height * kernel_width), mask=None, other=0.0).reshape(kernel_height, kernel_width)
                        
                        # Load weight slice for this input channel: [KH, KW]  
                        weight_slice = weights[:, c_in, :, :].reshape(n_offsets.shape[0], kernel_height, kernel_width)
                        
                        # Multiply and accumulate
                        for kh in range(kernel_height):
                            for kw in range(kernel_width):
                                input_vals = input_patch[kh, kw]
                                for i in range(n_offsets.shape[0]):
                                    acc[i] += input_vals * weight_slice[i, kh, kw]
                    
                    # Store output for this sequence position and output channel
                    output_offsets = (n_offsets * n_batch) + tl.arange(0, n_batch)
                    tl.store(output_base + output_offsets[:, None], acc[:, None], mask=(n_offsets < n_out_channels)[:, None])

@torch.fx.wrap
def fused_conv_flatten_transpose(input_tensor, conv_weight):
    # Get tensor dimensions
    n_batch, n_in_channels, in_height, in_width = input_tensor.shape
    n_out_channels, conv_in_channels, kernel_height, kernel_width = conv_weight.shape
    
    # Verify input channel compatibility
    assert n_in_channels == conv_in_channels, f"Input channels mismatch: {n_in_channels} vs {conv_in_channels}"
    
    # Calculate output dimensions based on stride
    stride_height = kernel_height  # ViT uses stride equal to kernel size
    stride_width = kernel_width
    out_height = (in_height - kernel_height) // stride_height + 1
    out_width = (in_width - kernel_width) // stride_width + 1
    output_seq_len = out_height * out_width
    
    # Create output tensor in sequence format [N, L, C_out]
    output = torch.empty((n_batch, output_seq_len, n_out_channels), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Choose optimal block sizes based on tensor dimensions
    BLOCK_SIZE_M = min(64, output_seq_len)  # Sequence dimension block size
    BLOCK_SIZE_N = min(32, n_out_channels)  # Channel dimension block size
    
    # Calculate grid dimensions
    grid = (output_seq_len, (n_out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    # Launch Triton kernel
    fused_conv_flatten_transpose_kernel[grid](
        input_tensor,
        conv_weight,
        output,
        n_batch,
        n_in_channels,
        in_height,
        in_width,
        n_out_channels,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    return fused_conv_flatten_transpose