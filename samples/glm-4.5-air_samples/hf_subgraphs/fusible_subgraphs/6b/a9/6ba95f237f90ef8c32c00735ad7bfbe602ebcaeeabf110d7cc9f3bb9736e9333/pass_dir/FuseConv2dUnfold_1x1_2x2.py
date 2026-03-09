import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Pattern matching: Conv2d + Unfold + Reshape
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_conv_unfold_kernel(
    weights_ptr,           # [128, 256, 1, 1]
    input_ptr,             # [1, 256, 32, 32]
    output_ptr,            # [1, 128, 4, 256]
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_channels,
    BLOCK_SIZE_M: tl.constexpr,  # output channels to process per program
    BLOCK_SIZE_N: tl.constexpr,  # input channels to process per program
    BLOCK_SIZE_H: tl.constexpr,  # output height dimension
    BLOCK_SIZE_W: tl.constexpr,  # output width dimension
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Each program handles BLOCK_SIZE_M output channels
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, out_channels)
    m_offs = m_start + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offs < out_channels
    
    # Conv2d parameters (1x1, stride 1, padding 0, dilation 1)
    kernel_size_h = 1
    kernel_size_w = 1
    stride_h = 1
    stride_w = 1
    padding_h = 0
    padding_w = 0
    dilation_h = 1
    dilation_w = 1
    
    # Calculate input coordinates for convolution
    # For output position (h_out, w_out), input is (h_out, w_out) due to stride=1, padding=0
    input_h_base = pid_h * 2  # stride=2 for unfold
    input_w_base = pid_w * 2
    
    # Process 4 spatial locations in the 2x2 patch at once
    # These are the 4 locations that will be extracted by unfold
    for h_patch in range(2):
        for w_patch in range(2):
            # Actual input coordinates for this patch element
            input_h = input_h_base + h_patch
            input_w = input_w_base + w_patch
            
            # Only process if within input bounds
            if input_h < in_height and input_w < in_width:
                # Process all output channels for this spatial location
                for m in range(m_start, m_end):
                    acc = 0.0
                    
                    # Process input channels in blocks
                    n_chunks = (in_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
                    for n_chunk in range(n_chunks):
                        n_start = n_chunk * BLOCK_SIZE_N
                        n_end = min((n_chunk + 1) * BLOCK_SIZE_N, in_channels)
                        n_offs = n_start + tl.arange(0, BLOCK_SIZE_N)
                        n_mask = n_offs < in_channels
                        
                        # Load weight for current output channel, input channel chunk
                        weights = tl.load(weights_ptr + m * in_channels + n_offs, 
                                        mask=n_mask, other=0.0)
                        
                        # Load input features for current spatial location, input channel chunk
                        input_offset = (0, n_offs, input_h, input_w)
                        inputs = tl.load(input_ptr + input_offset, 
                                       mask=n_mask, other=0.0)
                        
                        # Accumulate: 1x1 convolution is just weighted sum
                        acc += tl.sum(weights * inputs)
                    
                    # Store result at the right location in output
                    # Output layout: [1, 128, 4, 256]
                    # For each patch position (h_patch, w_patch), store column-wise
                    patch_idx = h_patch * 2 + w_patch
                    h_idx = pid_w  # Each program handles one column of patches
                    w_idx = pid_h  # Each program handles one row of patches
                    output_offset = (0, m_offs, patch_idx, h_idx * 256 + w_idx)
                    tl.store(output_ptr + output_offset, acc, mask=m_mask)

@torch.fx.wrap
def fused_conv_unforward_impl(weights, input_tensor):
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels = weights.shape[0]
    
    # Output after unfold: [1, 128, 4, 256]
    output_shape = (1, out_channels, 4, (in_height // 2 * in_width // 2))
    
    output = torch.zeros(output_shape, dtype=weights.dtype, device=weights.device)
    
    # Configure grid and block sizes
    BLOCK_SIZE_M = 128  # Process multiple output channels per program
    BLOCK_SIZE_N = 256  # Load multiple input channels per program
    BLOCK_SIZE_H = in_height // 2  # Output height patches
    BLOCK_SIZE_W = in_width // 2   # Output width patches
    
    grid = (
        (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,  # output channels
        (in_height + 1) // 2,  # output height patches (stride 2)
        (in_width + 1) // 2,   # output width patches (stride 2)
    )
    
    fused_conv_unfold_kernel[grid](
        weights_ptr=weights,
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return output

def replacement_func():
    return fused_conv_unforward_impl