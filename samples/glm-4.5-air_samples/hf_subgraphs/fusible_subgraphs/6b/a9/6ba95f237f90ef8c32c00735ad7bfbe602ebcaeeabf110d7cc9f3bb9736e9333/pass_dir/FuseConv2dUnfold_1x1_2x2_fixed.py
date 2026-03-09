import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Pattern matching: Conv2d + Unfold + Reshape
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
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
):
    pid_m = tl.program_id(0)
    pid_patch = tl.program_id(1)  # Process patches in column-major order
    
    # Each program handles BLOCK_SIZE_M output channels
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, out_channels)
    m_offs = m_start + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offs < out_channels
    
    # Decode patch position from pid_patch (256 patches total: 16x16)
    patch_h = pid_patch // 16
    patch_w = pid_patch % 16
    
    # Each patch has 4 positions for 2x2 kernel
    patch_positions = [
        (patch_h * 2, patch_w * 2),      # top-left
        (patch_h * 2, patch_w * 2 + 1),  # top-right
        (patch_h * 2 + 1, patch_w * 2),  # bottom-left
        (patch_h * 2 + 1, patch_w * 2 + 1) # bottom-right
    ]
    
    # Process the 4 positions in the patch
    for patch_idx, (input_h, input_w) in enumerate(patch_positions):
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
                    
                    # Load weights for current output channel, input channel chunk
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
                output_offset = (0, m_offs, patch_idx, pid_patch)
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
    
    # Grid: (output_channels, total_patches)
    grid = (
        (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,  # output channels
        in_height // 2 * in_width // 2,  # total patches (16x16 = 256)
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
    )
    
    return output

def replacement_func():
    return fused_conv_unforward_impl