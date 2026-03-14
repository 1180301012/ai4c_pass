import torch
import triton
import triton.language as tl

# Pattern matching function - matches cat + batch_norm + prelu only
# Pool and view remain outside the pattern
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    tmp_5 = torch.cat([in_5, in_6], 1)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_1, in_2, in_4, in_3, False, 0.1, 0.001)
    tmp_7 = torch.prelu(tmp_6, in_0)
    return tmp_7

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@triton.jit
def fused_cat_bn_prelu_kernel_channel(
    in5_ptr, in6_ptr,  # Input tensors to cat
    mean_ptr, var_ptr, weight_ptr, bias_ptr,  # BN params
    prelu_weight_ptr,  # PReLU weight
    out_ptr,  # Output
    channels: tl.constexpr,
    half_channels: tl.constexpr,
    spatial_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair - simpler logic for small batches
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Load BN parameters once per (batch, channel) pair
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    gamma = tl.load(weight_ptr + channel_idx)
    beta = tl.load(bias_ptr + channel_idx)
    prelu_w = tl.load(prelu_weight_ptr + channel_idx)
    
    # Precompute BN scale and shift
    inv_std = tl.rsqrt(var + eps)
    scale = gamma * inv_std
    shift = beta - mean * scale
    
    # Determine source channel  
    use_in6 = channel_idx >= half_channels
    src_channel = channel_idx - half_channels if channel_idx >= half_channels else channel_idx
    
    # Compute base pointers
    in_base = batch_idx * (half_channels * spatial_size) + src_channel * spatial_size
    out_base = batch_idx * (channels * spatial_size) + channel_idx * spatial_size
    
    # Select input pointer
    in_ptr = in6_ptr if channel_idx >= half_channels else in5_ptr
    
    # Process all spatial locations
    for s_start in range(0, spatial_size, BLOCK_SIZE):
        s_offsets = s_start + tl.arange(0, BLOCK_SIZE)
        mask = s_offsets < spatial_size
        
        # Load from input
        x = tl.load(in_ptr + in_base + s_offsets, mask=mask, other=0.0)
        
        # Apply BatchNorm and PReLU fused
        y = x * scale + shift
        y = tl.where(y >= 0, y, y * prelu_w)
        
        # Store
        tl.store(out_ptr + out_base + s_offsets, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_cat_bn_prelu_kernel_flat(
    in5_ptr, in6_ptr,  # Input tensors to cat
    mean_ptr, var_ptr, weight_ptr, bias_ptr,  # BN params
    prelu_weight_ptr,  # PReLU weight
    out_ptr,  # Output
    batch, 
    channels: tl.constexpr, 
    half_channels: tl.constexpr,
    spatial_size,
    total_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid over all elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decode flat index
    spatial_idx = offsets % spatial_size
    bc_idx = offsets // spatial_size
    channel_idx = bc_idx % channels
    batch_idx = bc_idx // channels
    
    # Load BN parameters
    mean = tl.load(mean_ptr + channel_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + channel_idx, mask=mask, other=1.0)
    gamma = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
    beta = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    prelu_w = tl.load(prelu_weight_ptr + channel_idx, mask=mask, other=0.25)
    
    # Precompute BN scale and shift
    inv_std = tl.rsqrt(var + eps)
    scale = gamma * inv_std
    shift = beta - mean * scale
    
    # Determine source channel
    use_in6 = channel_idx >= half_channels
    src_channel = tl.where(use_in6, channel_idx - half_channels, channel_idx)
    
    # Compute input offset
    in_offset = batch_idx * (half_channels * spatial_size) + src_channel * spatial_size + spatial_idx
    
    # Load from appropriate input
    x5 = tl.load(in5_ptr + in_offset, mask=mask & ~use_in6, other=0.0)
    x6 = tl.load(in6_ptr + in_offset, mask=mask & use_in6, other=0.0)
    x = tl.where(use_in6, x6, x5)
    
    # Apply BatchNorm
    y = x * scale + shift
    
    # Apply PReLU
    y = tl.where(y >= 0, y, y * prelu_w)
    
    # Compute output offset and store
    out_offset = batch_idx * (channels * spatial_size) + channel_idx * spatial_size + spatial_idx
    tl.store(out_ptr + out_offset, y, mask=mask)


@torch.fx.wrap
def fused_cat_bn_prelu(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Fused implementation of cat + batch_norm + prelu
    """
    batch = in_5.shape[0]
    half_channels = in_5.shape[1]  # 64
    channels = half_channels * 2   # 128
    height = in_5.shape[2]
    width = in_5.shape[3]
    spatial_size = height * width
    total_elements = batch * channels * spatial_size
    
    # Output tensor
    out = torch.empty((batch, channels, height, width), device=in_5.device, dtype=in_5.dtype)
    
    # For small batches, use channel-based parallelization (lower overhead)
    if batch <= 16:
        BLOCK_SIZE = 1024
        num_programs = batch * channels
        fused_cat_bn_prelu_kernel_channel[(num_programs,)](
            in_5, in_6,
            in_1, in_2, in_4, in_3,  # mean, var, weight, bias
            in_0,  # prelu weight
            out,
            channels, half_channels, spatial_size,
            eps=0.001,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # For larger batches, use flat parallelization with autotune
        BLOCK_SIZE_EST = 1024
        num_blocks = (total_elements + BLOCK_SIZE_EST - 1) // BLOCK_SIZE_EST
        fused_cat_bn_prelu_kernel_flat[(num_blocks,)](
            in_5, in_6,
            in_1, in_2, in_4, in_3,  # mean, var, weight, bias
            in_0,  # prelu weight
            out,
            batch, channels, half_channels, spatial_size, total_elements,
            eps=0.001,
        )
    
    return out


def replacement_func():
    return fused_cat_bn_prelu