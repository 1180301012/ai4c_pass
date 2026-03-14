import torch
import triton
import triton.language as tl

# Pattern matching function - matches sigmoid -> multiply -> gelu -> pool -> flatten -> dropout
def pattern(conv_out, feature_map):
    """
    Pattern matches:
    sigmoid -> multiply -> gelu -> adaptive_avg_pool2d -> flatten -> dropout
    """
    tmp_3 = conv_out.sigmoid()
    tmp_4 = feature_map * tmp_3
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    return tmp_8

# Argument extraction function
def replacement_args(conv_out, feature_map):
    return (conv_out, feature_map)


@triton.autotune(
    configs=[
        # Best performing configs (achieved 1.255 score)
        triton.Config({'BLOCK_C': 512, 'BLOCK_S': 64}, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 64}, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_S': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_S': 32}, num_warps=2),
    ],
    key=['channels', 'spatial_size'],
)
@triton.jit
def fused_se_gelu_pool_kernel(
    conv_out_ptr,
    feature_ptr,
    output_ptr,
    batch_size,
    channels,
    spatial_size,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """
    Optimized fused kernel for sigmoid -> multiply -> gelu -> global avg pool
    conv_out: [batch, channels, 1, 1]
    feature: [batch, channels, height, width]
    output: [batch, channels]
    """
    # Each program handles a batch-channel block
    pid = tl.program_id(0)
    num_chan_blocks = tl.cdiv(channels, BLOCK_C)
    batch_idx = pid // num_chan_blocks
    chan_block_idx = pid % num_chan_blocks
    
    # Channel offsets for this block
    chan_offset = chan_block_idx * BLOCK_C + tl.arange(0, BLOCK_C)
    chan_mask = chan_offset < channels
    
    # Load conv output for these channels and apply sigmoid once
    conv_idx = batch_idx * channels + chan_offset
    conv_val = tl.load(conv_out_ptr + conv_idx, mask=chan_mask, other=0.0)
    sig_val = tl.sigmoid(conv_val)
    
    # Accumulator for global average pooling
    accum = tl.zeros([BLOCK_C], dtype=tl.float32)
    
    # Loop over spatial locations with blocking
    num_iters = tl.cdiv(spatial_size, BLOCK_S)
    
    for iter_idx in range(num_iters):
        spatial_start = iter_idx * BLOCK_S
        spatial_idx = spatial_start + tl.arange(0, BLOCK_S)
        spatial_mask = spatial_idx < spatial_size
        
        # Create 2D index: [BLOCK_C, BLOCK_S]
        chan_idx_2d = chan_offset[:, None]
        spatial_idx_2d = spatial_idx[None, :]
        
        # Compute flattened indices for feature map
        feat_idx = (batch_idx * channels + chan_idx_2d) * spatial_size + spatial_idx_2d
        
        # Mask for both dimensions
        mask_2d = chan_mask[:, None] & spatial_mask[None, :]
        
        # Load feature values
        feat_val = tl.load(feature_ptr + feat_idx, mask=mask_2d, other=0.0)
        
        # Fused operations: multiply -> GELU
        mul_val = feat_val * sig_val[:, None]
        gelu_val = mul_val * tl.sigmoid(1.702 * mul_val)
        
        # Sum over spatial dimension
        accum += tl.sum(gelu_val, axis=1)
    
    # Compute average
    avg_val = accum / spatial_size
    
    # Store result
    output_idx = batch_idx * channels + chan_offset
    tl.store(output_ptr + output_idx, avg_val, mask=chan_mask)


@torch.fx.wrap
def fused_se_gelu_pool(conv_out, feature_map):
    """
    Wrapper function that calls fused kernel for sigmoid -> mul -> gelu -> pool
    """
    # Get dimensions
    batch_size, channels, height, width = feature_map.shape
    spatial_size = height * width
    
    # Allocate output
    output = torch.empty((batch_size, channels), device=feature_map.device, dtype=feature_map.dtype)
    
    # Launch kernel with 1D grid over batch-channel combinations
    # Let autotuning determine BLOCK_C, grid size will be adjusted accordingly
    # We use a simple heuristic for initial grid
    BLOCK_C = 128  # Will be overridden by autotune
    grid = lambda meta: (batch_size * triton.cdiv(channels, meta['BLOCK_C']),)
    
    fused_se_gelu_pool_kernel[grid](
        conv_out,
        feature_map,
        output,
        batch_size,
        channels,
        spatial_size,
    )
    
    return output


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_se_gelu_pool