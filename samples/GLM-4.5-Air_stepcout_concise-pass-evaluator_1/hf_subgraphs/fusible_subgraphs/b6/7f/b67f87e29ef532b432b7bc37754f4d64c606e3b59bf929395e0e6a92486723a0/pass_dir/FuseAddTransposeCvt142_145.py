import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match: addition + transpose + specific view for CVT-13-384 start142_end145_20"""
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.permute(0, 2, 1)
    tmp_2 = tmp_1.view(1, 192, 48, 48)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def add_transpose_spatial_kernel_192_48(
    x_ptr, y_ptr,
    out_ptr,
    n_batch, n_spatial_total, n_channels,
    spatial_width, spatial_height,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for [1, 2304, 192] -> [1, 192, 48, 48]"""
    pid = tl.program_id(0)
    
    # Each program handles a specific channel and spatial position
    batch_idx = pid // (n_channels * spatial_height * spatial_width)  # Should be 0 for single batch
    channel_idx = (pid // (spatial_height * spatial_width)) % n_channels
    flat_spatial_idx = pid % (spatial_height * spatial_width)
    
    # Map flat spatial index to 2D coordinates in transposed tensor
    h_idx = flat_spatial_idx // spatial_width
    w_idx = flat_spatial_idx % spatial_width
    
    # In original [1, 2304, 192]: [batch, feature, channel]
    # After permute(0, 2, 1): [1, 192, 2304] where we access as [batch, channel, feature]
    # The feature index after transpose corresponds to the original middle dimension
    feature_idx = h_idx * spatial_width + w_idx
    
    # Calculate input offset for original [1, 2304, 192] tensor
    input_offset = feature_idx * n_channels + channel_idx
    
    # Load corresponding elements from input tensors
    x_val = tl.load(x_ptr + input_offset, mask=input_offset < n_spatial_total * n_channels, other=0.0)
    y_val = tl.load(y_ptr + input_offset, mask=input_offset < n_spatial_total * n_channels, other=0.0)
    
    # Add elements
    sum_val = x_val + y_val
    
    # Calculate output offset in [1, 192, 48, 48] layout
    output_offset = batch_idx * (n_channels * spatial_height * spatial_width) + \
                   channel_idx * (spatial_height * spatial_width) + \
                   flat_spatial_idx
    
    # Store result in output tensor
    tl.store(out_ptr + output_offset, sum_val)

@torch.fx.wrap
def fused_add_transpose_192_48(x, y):
    """Fused operation for CVT-13-384 start142_end145_20 pattern"""
    # Input: [1, 2304, 192] -> Output: [1, 192, 48, 48]
    output_shape = (1, 192, 48, 48)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    n_batch, n_features, n_channels = x.shape
    n_spatial_total = n_features  # 2304
    spatial_width = 48  # 48 x 48 = 2304
    spatial_height = 48
    
    # Calculate grid size: one thread per channel per spatial position
    num_programs = n_batch * n_channels * spatial_height * spatial_width
    BLOCK_SIZE = 1  # Each thread handles one element
    
    # Call fused kernel
    add_transpose_spatial_kernel_192_48[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out.view(-1),  # Flatten for kernel
        n_batch=n_batch,
        n_spatial_total=n_spatial_total,
        n_channels=n_channels,
        spatial_width=spatial_width,
        spatial_height=spatial_height,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_add_transpose_192_48