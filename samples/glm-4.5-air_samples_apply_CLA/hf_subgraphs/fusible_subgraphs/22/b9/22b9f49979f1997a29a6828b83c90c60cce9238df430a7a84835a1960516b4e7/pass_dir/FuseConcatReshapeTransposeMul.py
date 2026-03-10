import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat([in_0, in_1, in_2], dim=1)
    tmp_1 = tmp_0.reshape(1, 8, -1, -1)
    tmp_2 = tmp_1.transpose(-1, -2)
    tmp_3 = in_3 * tmp_2
    tmp_4 = torch.nn.functional.pad(tmp_3, (0, 0, 1, 0, 0, 0), 'constant', None)
    return (tmp_4,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, 
    out_ptr,
    in_0_stride_0, in_0_stride_1, in_0_stride_2, in_0_stride_3,
    in_1_stride_0, in_1_stride_1, in_1_stride_2, in_1_stride_3,
    in_2_stride_0, in_2_stride_1, in_2_stride_2, in_2_stride_3,
    in_3_stride_0, in_3_stride_1, in_3_stride_2, in_3_stride_3,
    out_stride_0, out_stride_1, out_stride_2, out_stride_3,
    n_channels_0, n_channels_1, n_channels_2,
    height, width,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Compute program_id for output tensor dimensions
    pid_m = tl.program_id(0)  # batch size (always 1)
    pid_k = tl.program_id(1)  # feature dim (always 8)
    pid_n = tl.program_id(2)  # spatial dim (width or height)
    
    # Compute base pointer for position in output
    base_offset = pid_m * out_stride_0 + pid_k * out_stride_1 + pid_n * out_stride_2
    
    # Load from in_3 (the multiplier tensor)
    in_3_val = tl.load(in_3_ptr + base_offset, other=0.0)
    
    # Compute which input tensor and local position based on channel dimension
    total_channels = n_channels_0 + n_channels_1 + n_channels_2
    channel_offset = pid_n * height * total_channels  # spatial position * H * total_channels
    local_channel = tl.program_id(3)  # channel mapping program
    
    # Determine which input tensor and local channel within that tensor
    if local_channel < n_channels_0:
        base_ptr = in_0_ptr
        base_offset_0 = local_channel * in_0_stride_1 + channel_offset
        val = tl.load(base_ptr + base_offset_0, other=0.0)
    elif local_channel < n_channels_0 + n_channels_1:
        base_ptr = in_1_ptr
        local_ch = local_channel - n_channels_0
        base_offset_0 = local_ch * in_1_stride_1 + channel_offset
        val = tl.load(base_ptr + base_offset_0, other=0.0)
    else:
        base_ptr = in_2_ptr
        local_ch = local_channel - n_channels_0 - n_channels_1
        base_offset_0 = local_ch * in_2_stride_1 + channel_offset
        val = tl.load(base_ptr + base_offset_0, other=0.0)
    
    # Apply multiplication and transpose semantics
    # We're computing: result = in_3_val * val, then we'll handle transpose in offsets
    result = in_3_val * val
    
    # Store with transpose semantics (swapping height and width dimensions)
    out_offset_0 = pid_m * out_stride_0 + pid_k * out_stride_1 + pid_n * out_stride_2
    if pid_n < K:  # Only store if within valid bounds
        # For the padded output, we need to handle the padding
        if pid_n == K:  # This is the padding dimension
            # Store zeros for padding
            tl.store(out_ptr + out_offset_0, 0.0)
        else:
            tl.store(out_ptr + out_offset_0, result)

@torch.fx.wrap
def fused_computation(in_0, in_1, in_2, in_3):
    # Compute output shape based on in_3 (after transpose + pad)
    out_shape = list(in_3.shape)
    out_shape[-2] = out_shape[-2] + 1  # Add padding to height dimension
    
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Extract tensor information
    batch_size, n_channels_0, height_0, width_0 = in_0.shape
    _, n_channels_1, height_1, width_1 = in_1.shape
    _, n_channels_2, height_2, width_2 = in_2.shape
    
    # All inputs should have same H, W due to concat
    height, width = height_0, width_0
    
    # Get tensor strides
    in_0_strides = in_0.stride()
    in_1_strides = in_1.stride()
    in_2_strides = in_2.stride()
    in_3_strides = in_3.stride()
    out_strides = out.stride()
    
    # Launch kernel
    total_channels = n_channels_0 + n_channels_1 + n_channels_2
    
    # Grid setup: batch, features, spatial_dim, channel_mapping
    grid_launch = lambda meta: (
        out.shape[0],  # batch size
        out.shape[1],  # features (8) 
        out.shape[2],  # spatial dimension
        (total_channels + 127) // 128,  # channel mapping blocks
    )
    
    fused_kernel[grid_launch](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        in_0_stride_0=in_0_strides[0], in_0_stride_1=in_0_strides[1], 
        in_0_stride_2=in_0_strides[2], in_0_stride_3=in_0_strides[3],
        in_1_stride_0=in_1_strides[0], in_1_stride_1=in_1_strides[1],
        in_1_stride_2=in_1_strides[2], in_1_stride_3=in_1_strides[3],
        in_2_stride_0=in_2_strides[0], in_2_stride_1=in_2_strides[1],
        in_2_stride_2=in_2_strides[2], in_2_stride_3=in_2_strides[3],
        in_3_stride_0=in_3_strides[0], in_3_stride_1=in_3_strides[1],
        in_3_stride_2=in_3_strides[2], in_3_stride_3=in_3_strides[3],
        out_stride_0=out_strides[0], out_stride_1=out_strides[1],
        out_stride_2=out_strides[2], out_stride_3=out_strides[3],
        n_channels_0=n_channels_0, n_channels_1=n_channels_1, n_channels_2=n_channels_2,
        height=height, width=width,
        M=8, K=out.shape[2], N=out.shape[3],
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32
    )
    
    return out

def replacement_func():
    return fused_computation