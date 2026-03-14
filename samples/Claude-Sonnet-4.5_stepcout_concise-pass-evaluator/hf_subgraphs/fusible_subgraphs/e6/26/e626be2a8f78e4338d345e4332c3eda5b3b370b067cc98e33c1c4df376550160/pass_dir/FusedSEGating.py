import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Match the SE gating pattern: conv2d + hard_sigmoid + multiply"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2 + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return (tmp_6,)

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for replacement function"""
    return (in_0, in_1, in_2, in_3)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_IC': 32, 'BLOCK_SIZE_SPATIAL': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_IC': 64, 'BLOCK_SIZE_SPATIAL': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_IC': 32, 'BLOCK_SIZE_SPATIAL': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_IC': 64, 'BLOCK_SIZE_SPATIAL': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_IC': 64, 'BLOCK_SIZE_SPATIAL': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_IC': 128, 'BLOCK_SIZE_SPATIAL': 128}, num_warps=4),
    ],
    key=['in_channels'],
)
@triton.jit
def se_gating_kernel(
    # Input pointers
    feature_map_ptr,  # in_2: [batch, out_channels, H, W]
    excitation_ptr,   # in_3: [batch, in_channels, 1, 1]
    weight_ptr,       # in_1: [out_channels, in_channels, 1, 1]
    bias_ptr,         # in_0: [out_channels]
    # Output pointer
    output_ptr,       # [batch, out_channels, H, W]
    # Dimensions
    batch_size, in_channels, out_channels, spatial_size,
    # Strides
    fm_batch_stride, fm_ch_stride,
    exc_batch_stride, exc_ch_stride,
    w_out_stride, w_in_stride,
    # Block configuration
    BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    """
    Fused SE gating kernel.
    Grid: (batch * out_channels,)
    """
    pid = tl.program_id(0)
    
    batch = pid // out_channels
    channel = pid % out_channels
    
    # Compute gate value using vectorized reduction over input channels
    gate = 0.0
    for ic_start in range(0, in_channels, BLOCK_SIZE_IC):
        ic_offsets = ic_start + tl.arange(0, BLOCK_SIZE_IC)
        ic_mask = ic_offsets < in_channels
        
        # Load excitation[batch, ic, 0, 0] and weight[channel, ic, 0, 0]
        exc_offsets = batch * exc_batch_stride + ic_offsets * exc_ch_stride
        exc_vals = tl.load(excitation_ptr + exc_offsets, mask=ic_mask, other=0.0)
        
        w_offsets = channel * w_out_stride + ic_offsets * w_in_stride
        w_vals = tl.load(weight_ptr + w_offsets, mask=ic_mask, other=0.0)
        
        # Accumulate dot product
        gate += tl.sum(w_vals * exc_vals)
    
    # Add bias and apply hard sigmoid: clamp((x + 1) / 2, 0, 1)
    bias_val = tl.load(bias_ptr + channel)
    gate = (gate + bias_val + 1.0) / 2.0
    gate = tl.maximum(0.0, tl.minimum(1.0, gate))
    
    # Base offset for this (batch, channel) pair
    base_offset = batch * fm_batch_stride + channel * fm_ch_stride
    
    # Process all spatial positions in vectorized blocks
    for spatial_start in range(0, spatial_size, BLOCK_SIZE_SPATIAL):
        spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE_SPATIAL)
        spatial_mask = spatial_offsets < spatial_size
        
        # Compute feature map offsets (assuming contiguous spatial dimension)
        fm_offsets = base_offset + spatial_offsets
        
        # Load, multiply by gate, and store
        fm_vals = tl.load(feature_map_ptr + fm_offsets, mask=spatial_mask, other=0.0)
        output_vals = fm_vals * gate
        tl.store(output_ptr + fm_offsets, output_vals, mask=spatial_mask)

@torch.fx.wrap
def fused_se_gating(in_0, in_1, in_2, in_3):
    """
    Wrapper function for the fused SE gating kernel.
    
    Args:
        in_0: bias [out_channels]
        in_1: weight [out_channels, in_channels, 1, 1]
        in_2: feature_map [batch, out_channels, H, W]
        in_3: excitation [batch, in_channels, 1, 1]
    
    Returns:
        output: [batch, out_channels, H, W]
    """
    # Extract dimensions
    batch_size, out_channels, height, width = in_2.shape
    _, in_channels, _, _ = in_3.shape
    spatial_size = height * width
    
    # Make feature map contiguous for efficient spatial access
    in_2_contig = in_2.contiguous()
    
    # Allocate output tensor
    output = torch.empty_like(in_2_contig)
    
    # Configure grid - one program per (batch, channel) pair
    grid = (batch_size * out_channels,)
    
    # Launch kernel (autotuning will select best BLOCK_SIZE_IC and BLOCK_SIZE_SPATIAL)
    se_gating_kernel[grid](
        in_2_contig, in_3, in_1, in_0, output,
        batch_size, in_channels, out_channels, spatial_size,
        in_2_contig.stride(0), in_2_contig.stride(1),
        in_3.stride(0), in_3.stride(1),
        in_1.stride(0), in_1.stride(1),
    )
    
    return output

def replacement_func():
    """Return the replacement function (not called)"""
    return fused_se_gating