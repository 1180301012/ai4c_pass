import torch
import triton
import triton.language as tl


def pattern(conv_output, feature_map):
    """
    Pattern matching for post-conv SE attention operations:
    hardsigmoid -> multiply -> adaptive_avg_pool2d -> flatten -> dropout
    """
    tmp_3 = torch.nn.functional.hardsigmoid(conv_output, False)
    tmp_4 = feature_map * tmp_3
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(conv_output, feature_map):
    return (conv_output, feature_map)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['spatial_size'],
)
@triton.jit
def fused_se_attention_kernel(
    se_input_ptr,  # Output from conv2d [B, C, 1, 1]
    feature_ptr,   # Input feature map [B, C, H, W]
    output_ptr,    # Output [B, C]
    batch_size,
    num_channels,
    spatial_size,  # H * W
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: hardsigmoid -> multiply -> global avg pool
    """
    pid = tl.program_id(0)
    
    batch_idx = pid // num_channels
    channel_idx = pid % num_channels
    
    # Load and apply hardsigmoid to SE value
    se_idx = batch_idx * num_channels + channel_idx
    se_val = tl.load(se_input_ptr + se_idx)
    se_val = tl.maximum(0.0, tl.minimum(1.0, (se_val + 3.0) / 6.0))
    
    # Base address for this channel's features
    feature_base = (batch_idx * num_channels + channel_idx) * spatial_size
    
    # Accumulate weighted sum
    acc = 0.0
    
    # Unrolled loop for better performance
    num_iters = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(num_iters):
        offset = i * BLOCK_SIZE
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        feature_vals = tl.load(feature_ptr + feature_base + offsets, mask=mask, other=0.0)
        acc += tl.sum(tl.where(mask, feature_vals * se_val, 0.0))
    
    # Store average
    tl.store(output_ptr + se_idx, acc / spatial_size)


@torch.fx.wrap
def fused_se_attention(conv_output, feature_map):
    """
    Optimized implementation of SE attention block post-conv operations.
    """
    # Get dimensions
    B, C, H, W = feature_map.shape
    spatial_size = H * W
    
    # Allocate output
    output = torch.empty((B, C), device=feature_map.device, dtype=feature_map.dtype)
    
    # Launch kernel
    total_elements = B * C
    grid = (total_elements,)
    
    fused_se_attention_kernel[grid](
        conv_output,
        feature_map,
        output,
        B,
        C,
        spatial_size,
    )
    
    return output


def replacement_func():
    return fused_se_attention