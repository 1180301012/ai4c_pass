import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Pattern matching: concat + reshape + transpose + multiply + pad"""
    tmp_0 = torch.cat([in_0, in_1, in_2], dim=1)
    tmp_1 = tmp_0.reshape(1, 8, -1, tmp_0.shape[-1] * tmp_0.shape[-2] // 8)
    tmp_2 = tmp_1.transpose(-1, -2)
    tmp_3 = in_3 * tmp_2
    tmp_4 = torch.nn.functional.pad(tmp_3, (0, 0, 1, 0, 0, 0), 'constant', None)
    return (tmp_4,)

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the fusion kernel"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    in_0_shape0, in_0_shape1, in_0_shape2, in_0_shape3,
    in_1_shape0, in_1_shape1, in_1_shape2, in_1_shape3,
    in_2_shape0, in_2_shape1, in_2_shape2, in_2_shape3,
    out_shape0, out_shape1, out_shape2, out_shape3,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused kernel that handles concat + reshape + transpose + multiply"""
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute total concatenated channels
    total_channels = in_0_shape1 + in_1_shape1 + in_2_shape1
    dim_c_per_group = total_channels // 8
    spatial_dim = in_0_shape2 * in_0_shape3
    
    # Create output tensor indices
    # Output shape: [1, 8, H*W+1, dim_c_per_group]
    # We'll compute one element per program for simplicity
    
    # Flatten the output indices excluding the batch and padding dimensions
    total_output_elements = out_shape2 * out_shape3  # (H*W+1) * dim_c_per_group
    
    if pid >= total_output_elements:
        return
    
    # Unflatten the index to get [spatial_idx, c_group_idx]
    spatial_idx = pid // dim_c_per_group
    c_group_idx = pid % dim_c_per_group
    
    # Determine which batch element this is (always 0 since we have only one batch)
    batch_idx = 0
    
    # Calculate the source tensor information
    # The concatenated tensor is [1, total_channels, H, W]
    # Reshape to [1, 8, H*W, total_channels//8] then transpose to [1, 8, total_channels//8, H*W]
    # So element [0, c_group_idx, spatial_idx] comes from:
    # Concatenated tensor position: [0, c_group_idx * 8 + local_channel, spatial_idx // W, spatial_idx % W]
    
    # Get the original channel in the concatenated tensor
    original_channel = c_group_idx * 8 + 0  # We're processing one channel at a time
    
    # Determine which input tensor this channel comes from
    if original_channel < in_0_shape1:
        # From in_0
        src_channel = original_channel
        tensor_ptr = in_0_ptr
        src_shape1 = in_0_shape1
        src_shape2 = in_0_shape2
        src_shape3 = in_0_shape3
    elif original_channel < in_0_shape1 + in_1_shape1:
        # From in_1
        src_channel = original_channel - in_0_shape1
        tensor_ptr = in_1_ptr
        src_shape1 = in_1_shape1
        src_shape2 = in_1_shape2
        src_shape3 = in_1_shape3
    else:
        # From in_2
        src_channel = original_channel - in_0_shape1 - in_1_shape1
        tensor_ptr = in_2_ptr
        src_shape1 = in_2_shape1
        src_shape2 = in_2_shape2
        src_shape3 = in_2_shape3
    
    # Load from source tensor
    src_offset = batch_idx * (src_shape1 * src_shape2 * src_shape3) + \
                src_channel * (src_shape2 * src_shape3) + \
                spatial_idx
    src_value = tl.load(tensor_ptr + src_offset, mask=spatial_idx < spatial_dim, other=0.0)
    
    # Load from in_3 tensor
    in_3_offset = batch_idx * (out_shape1 * out_shape2 * out_shape3) + \
                  c_group_idx * out_shape3 + \
                  spatial_idx
    in_3_value = tl.load(in_3_ptr + in_3_offset, mask=spatial_idx < spatial_dim, other=0.0)
    
    # Multiply
    result = src_value * in_3_value
    
    # Store to output
    out_offset = batch_idx * (out_shape1 * out_shape2 * out_shape3) + \
                c_group_idx * out_shape3 + \
                spatial_idx
    tl.store(out_ptr + out_offset, result, mask=spatial_idx < spatial_dim)

@torch.fx.wrap
def fused_computation(in_0, in_1, in_2, in_3):
    """Wrapper function for the fused computation"""
    
    # Calculate output shapes
    total_channels = in_0.shape[1] + in_1.shape[1] + in_2.shape[1]
    assert total_channels % 8 == 0, "Total channels must be divisible by 8"
    
    spatial_dim = in_0.shape[2] * in_0.shape[3]
    dim_c_per_group = total_channels // 8
    
    # The output after reshape + transpose + multiply has shape [1, 8, spatial_dim, dim_c_per_group] = [1, 8, H*W, dim_c_per_group]
    # After padding: [1, 8, spatial_dim+1, dim_c_per_group] = [1, 8, H*W+1, dim_c_per_group]
    out_shape = (in_0.shape[0], 8, spatial_dim + 1, dim_c_per_group)
    
    # Create output tensor
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Get pointer data
    in_0_ptr = in_0.data_ptr()
    in_1_ptr = in_1.data_ptr()  
    in_2_ptr = in_2.data_ptr()
    in_3_ptr = in_3.data_ptr()
    out_ptr = out.data_ptr()
    
    # Launch kernel with 1D grid, one program per output element
    total_output_elements = (spatial_dim + 1) * dim_c_per_group
    grid_size = (total_output_elements,)
    
    fused_kernel[grid_size](
        in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
        in_0.shape[0], in_0.shape[1], in_0.shape[2], in_0.shape[3],
        in_1.shape[0], in_1.shape[1], in_1.shape[2], in_1.shape[3], 
        in_2.shape[0], in_2.shape[1], in_2.shape[2], in_2.shape[3],
        out_shape[0], out_shape[1], out_shape[2], out_shape[3],
        0, 0  # Block sizes not needed for 1D kernel
    )
    
    return out

def replacement_func():
    """Return the fused computation function"""
    return fused_computation