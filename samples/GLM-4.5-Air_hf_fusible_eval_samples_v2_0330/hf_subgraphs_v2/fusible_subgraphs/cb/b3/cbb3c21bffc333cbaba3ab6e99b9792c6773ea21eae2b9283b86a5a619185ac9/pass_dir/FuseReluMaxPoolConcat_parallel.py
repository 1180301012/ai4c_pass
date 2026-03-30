import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    return tmp_0, tmp_1, tmp_2, tmp_3


def replacement_args(in_0):
    return (in_0,)


# Triton kernel that fuses ReLU + 3x max_pool2d + concatenation
@triton.jit
def fused_relu_maxpool_concat_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles BLOCK_SIZE_M * BLOCK_SIZE_N output elements
    batch_idx = tl.program_id(0)
    elem_idx = tl.program_id(1)
    
    # Calculate which output channels and spatial positions this program handles
    out_channel_start = elem_idx * BLOCK_SIZE_M
    spatial_start = (elem_idx // ((out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)) * BLOCK_SIZE_N
    
    # Create masks for valid indices
    out_channel_mask = out_channel_start + tl.arange(0, BLOCK_SIZE_M) < out_channels
    spatial_mask = spatial_start + tl.arange(0, BLOCK_SIZE_N) < height * width
    
    # Calculate global indices
    out_channel = out_channel_start + tl.arange(0, BLOCK_SIZE_M)[:, None]
    spatial_pos = spatial_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
    
    # Determine which part of the concatenated output this corresponds to
    # (0=ReLU, 1-3=max_pools) and map back to input channel
    output_part = out_channel // n_channels
    input_channel = out_channel % n_channels
    
    # Calculate input and output offsets
    input_offset = (batch_idx * n_channels * height * width + 
                   input_channel * height * width + spatial_pos)
    output_offset = (batch_idx * out_channels * height * width + 
                    out_channel * height * width + spatial_pos)
    
    # Load input data
    input_data = tl.load(input_ptr + input_offset, 
                        mask=out_channel_mask[:, None] & spatial_mask[None, :])
    
    # For ReLU part (output_part == 0), store directly
    relu_mask = output_part == 0
    relu_out = tl.maximum(input_data, 0.0)
    
    if tl.any(relu_mask):
        relu_output_offset = output_offset[relu_mask]
        relu_data = relu_out[relu_mask]
        tl.store(output_ptr + relu_output_offset, relu_data,
                mask=out_channel_mask[relu_mask][:, None] & spatial_mask[None, :])
    
    # For each max_pool part, compute 5x5 max pooling
    for pool_part in range(1, 4):
        pool_mask = output_part == pool_part
        if not tl.any(pool_mask):
            continue
            
        # Initialize max pooling output
        pool_out = tl.full([BLOCK_SIZE_M, BLOCK_SIZE_N], -float('inf'), dtype=input_data.dtype)
        
        # 5x5 max pooling with stride 1, padding 2
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                # Calculate neighbor spatial position with padding
                neighbor_spatial = spatial_pos + dy * width + dx
                spatial_valid = (neighbor_spatial >= 0) & (neighbor_spatial < height * width)
                
                if tl.any(spatial_valid):
                    neighbor_offset = (batch_idx * n_channels * height * width + 
                                     input_channel * height * width + 
                                     neighbor_spatial)
                    
                    neighbor_data = tl.load(input_ptr + neighbor_offset,
                                          mask=pool_mask[:, None] & spatial_valid[None, :],
                                          other=-float('inf'))
                    
                    pool_out = tl.maximum(pool_out, neighbor_data)
        
        # Store max pool output
        pool_output_offset = output_offset[pool_mask]
        tl.store(output_ptr + pool_output_offset, pool_out,
                mask=out_channel_mask[pool_mask][:, None] & spatial_mask[None, :])


@torch.fx.wrap
def fused_relu_maxpool_concat(in_0):
    # Get input tensor properties
    batch_size, n_channels, height, width = in_0.shape
    dtype = in_0.dtype
    
    # Create output tensor: concatenated along channel dimension (4x channels)
    out_channels = n_channels * 4
    output = torch.empty((batch_size, out_channels, height, width), dtype=dtype, device=in_0.device)
    
    # Optimize block sizes based on tensor sizes
    BLOCK_SIZE_M = 64  # Process 64 output channels at a time
    BLOCK_SIZE_N = 128  # Process 128 spatial elements at a time
    
    # Calculate grid dimensions
    per_batch_elements = out_channels * height * width
    num_elements = (per_batch_elements + BLOCK_SIZE_M * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N)
    
    # Launch Triton kernel
    fused_relu_maxpool_concat_kernel[(batch_size, num_elements)](
        in_0,
        output,
        n_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output


def replacement_func():
    return fused_relu_maxpool_concat