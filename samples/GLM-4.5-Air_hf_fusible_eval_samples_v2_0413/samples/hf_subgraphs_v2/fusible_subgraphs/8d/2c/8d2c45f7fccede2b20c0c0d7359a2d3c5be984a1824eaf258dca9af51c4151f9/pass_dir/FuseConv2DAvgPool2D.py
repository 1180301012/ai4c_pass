import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern matching: conv2d followed by avg_pool2d"""
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    return tmp_2

def replacement_args(in_0, in_1):
    """Extract arguments for the fused conv2d + avg_pool2d operation"""
    return (in_0, in_1)

@triton.jit
def fused_conv_avg_pool_kernel(
    input_ptr,           # Input tensor [B, C_in, H, W]
    weight_ptr,          # Weight tensor [C_out, C_in, 1, 1]
    output_ptr,          # Output tensor [B, C_out, H_out, W_out]
    num_batches: tl.constexpr,
    num_channels_in: tl.constexpr,
    num_channels_out: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused conv2d + avg_pool2d kernel"""
    
    # Program ID for batch and channel parallelism
    batch_id = tl.program_id(0)
    channel_out_id = tl.program_id(1)
    
    # Calculate base positions
    batch_offset = batch_id * num_channels_in * input_height * input_width
    channel_out_offset = channel_out_id * (input_height // 2) * (input_width // 2)
    
    # Loop over input channels and output spatial positions
    for c_in in tl.range(0, num_channels_in, BLOCK_SIZE_K):
        for h_out in tl.range(0, output_height, BLOCK_SIZE_M):
            for w_out in tl.range(0, output_width, BLOCK_SIZE_N):
                
                # Calculate pointers for this block
                in_base = batch_offset + (c_in + h_out * 2) * input_width + w_out * 2
                weight_base = (channel_out_id * num_channels_in + c_in) * 1 * 1
                out_base = batch_offset + channel_out_offset + (h_out * output_width + w_out)
                
                # Initialize accumulator
                acc = 0.0
                
                # Read weights (1x1 kernel, so single value)
                weight_val = tl.load(weight_ptr + weight_base)
                
                # Read input pixels (2x2 window for avg pool)
                for h_idx in range(2):
                    for w_idx in range(2):
                        in_offset = in_base + h_idx * input_width + w_idx
                        pixel_val = tl.load(input_ptr + in_offset)
                        acc += pixel_val * weight_val
                
                # Average and store result
                acc = acc / 4.0
                tl.store(output_ptr + out_base, acc)

@torch.fx.wrap
def fused_conv_avg_pool(in_0, in_1):
    """Wrapper for fused conv2d + avg_pool2d operation"""
    
    # Get input shapes
    B, C_in, H_in, W_in = in_1.shape
    C_out, _, _, _ = in_0.shape
    
    # Calculate output shape after conv2d (no padding, stride 1) + avg_pool2d (stride 2)
    H_out = (H_in // 2) 
    W_out = (W_in // 2)
    
    # Create output tensor
    output = torch.empty((B, C_out, H_out, W_out), dtype=in_1.dtype, device=in_1.device)
    
    # Block size configuration for optimal GPU occupancy
    BLOCK_SIZE_M = 8   # Output height dimension
    BLOCK_SIZE_N = 8   # Output width dimension
    BLOCK_SIZE_K = 32  # Input channel dimension
    
    # Calculate grid dimensions
    grid_z = B  # Batch dimension
    grid_y = C_out  # Output channels dimension
    grid_x = ((H_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * ((W_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    # Launch kernel
    fused_conv_avg_pool_kernel[(grid_z, grid_y, grid_x)](
        input_ptr=in_1,
        weight_ptr=in_0,
        output_ptr=output,
        num_batches=B,
        num_channels_in=C_in,
        num_channels_out=C_out,
        input_height=H_in,
        input_width=W_in,
        output_height=H_out,
        output_width=W_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    """Return the fused function"""
    return fused_conv_avg_pool