import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern matching: Conv2D followed by Flatten"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def conv2d_flatten_kernel(
    bias_ptr, out_ptr,
    weight_ptr, in_ptr,
    N, C_out, C_in, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Conv2D + Flatten kernel using Triton"""
    batch_idx = tl.program_id(0)
    out_idx = tl.program_id(1)
    
    # Calculate out indices from flattened output [N, C_out, -1]
    flattened_size = H * W
    base_offset = batch_idx * C_out * flattened_size + out_idx * flattened_size
    
    # Load bias
    bias = tl.load(bias_ptr + out_idx)
    
    # Conv2D computation for each spatial position
    for h_idx in range(0, H):
        for w_idx in range(0, W):
            spatial_offset = h_idx * W + w_idx
            out_offset = base_offset + spatial_offset
            
            # Accumulate result with bias
            result = bias
            
            # 1x1 convolution: multiply kernel values with input values
            channel_offset = batch_idx * C_in * H * W + out_idx
            result = result + tl.load(weight_ptr + out_idx)
            
            # Complete final output
            final_result = result + tl.load(in_ptr + base_offset + spatial_offset)
            
            # Store result directly to flattened output
            tl.store(out_ptr + out_offset, final_result)

@torch.fx.wrap
def conv2d_flatten(in_0, in_1, in_2):
    """Fused Conv2D + Flatten implementation"""
    # Get input shapes
    N, C_in, H, W = in_2.shape
    C_out = in_1.shape[0]
    
    # Output shape after flatten: [N, C_out, H*W]
    out_size = N * C_out * H * W
    
    # Allocate output tensor
    out = torch.empty((N, C_out, H * W), dtype=in_2.dtype, device=in_2.device)
    
    # Block size for Triton kernel
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    num_batches = N
    num_output_channels = C_out
    num_spatial_elements = H * W
    num_programs = (num_spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Handle different batch sizes efficiently
    if num_spatial_elements <= BLOCK_SIZE:
        # Single program per batch-channel combination
        for batch in range(num_batches):
            for channel in range(num_output_channels):
                conv2d_flatten_kernel[(1,)](
                    bias_ptr=in_0,
                    out_ptr=out,
                    weight_ptr=in_1,
                    in_ptr=in_2,
                    N=num_batches,
                    C_out=num_output_channels,
                    C_in=C_in,
                    H=H,
                    W=W,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
    else:
        # Multiple programs per batch-channel combination
        grid = (num_batches, num_output_channels, num_programs)
        conv2d_flatten_kernel[grid](
            bias_ptr=in_0,
            out_ptr=out,
            weight_ptr=in_1,
            in_ptr=in_2,
            N=num_batches,
            C_out=num_output_channels,
            C_in=C_in,
            H=H,
                    W=W,
                    BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

def replacement_func():
    """Return the fused function"""
    return conv2d_flatten