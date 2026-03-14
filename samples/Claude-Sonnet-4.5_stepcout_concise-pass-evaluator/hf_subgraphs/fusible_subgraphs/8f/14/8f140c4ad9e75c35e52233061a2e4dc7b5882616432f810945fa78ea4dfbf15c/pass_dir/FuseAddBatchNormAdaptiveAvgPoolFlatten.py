import torch
import triton
import triton.language as tl


def pattern(conv_out, residual, running_mean, running_var, bn_weight, bn_bias):
    """
    Pattern to match: add + batch_norm + adaptive_avg_pool2d + flatten
    Matching the memory-bound operations after conv2d
    """
    added = residual + conv_out
    bn_out = torch.nn.functional.batch_norm(added, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    pooled = torch.nn.functional.adaptive_avg_pool2d(bn_out, 1)
    flattened = pooled.flatten(1, -1)
    return flattened


def replacement_args(conv_out, residual, running_mean, running_var, bn_weight, bn_bias):
    return (conv_out, residual, running_mean, running_var, bn_weight, bn_bias)


@triton.autotune(
    configs=[
        # Larger blocks for large batches
        triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_C': 256, 'BLOCK_SIZE_HW': 64}, num_warps=8),
        # Medium blocks
        triton.Config({'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_HW': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_HW': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_HW': 64}, num_warps=4),
        # Smaller blocks for small batches (better GPU utilization)
        triton.Config({'BLOCK_SIZE_C': 16, 'BLOCK_SIZE_HW': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_C': 16, 'BLOCK_SIZE_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_HW': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE_C': 16, 'BLOCK_SIZE_HW': 16}, num_warps=1),
    ],
    key=['C', 'HW'],
)
@triton.jit
def fused_add_bn_pool_kernel(
    conv_out_ptr,
    residual_ptr,
    running_mean_ptr,
    running_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    output_ptr,
    B, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    """
    Fused kernel for: add + batchnorm + adaptive_avg_pool2d + flatten
    
    Each program handles one batch and one channel block.
    """
    batch_idx = tl.program_id(0)
    c_block_idx = tl.program_id(1)
    
    # Channel offsets
    c_offsets = c_block_idx * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    
    # Load batch norm parameters once
    running_mean = tl.load(running_mean_ptr + c_offsets, mask=c_mask, other=0.0)
    running_var = tl.load(running_var_ptr + c_offsets, mask=c_mask, other=0.0)
    bn_weight = tl.load(bn_weight_ptr + c_offsets, mask=c_mask, other=1.0)
    bn_bias = tl.load(bn_bias_ptr + c_offsets, mask=c_mask, other=0.0)
    
    # Compute batch norm scale and shift
    scale = bn_weight / tl.sqrt(running_var + eps)
    shift = bn_bias - running_mean * scale
    
    # Accumulator for adaptive average pooling
    accumulator = tl.zeros([BLOCK_SIZE_C], dtype=tl.float32)
    
    # Process spatial dimensions in blocks
    num_hw_blocks = tl.cdiv(HW, BLOCK_SIZE_HW)
    for hw_block_idx in range(num_hw_blocks):
        hw_offsets = hw_block_idx * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
        hw_mask = hw_offsets < HW
        
        # Create 2D indexing: [BLOCK_SIZE_C, BLOCK_SIZE_HW]
        c_idx = c_offsets[:, None]
        hw_idx = hw_offsets[None, :]
        
        # Compute flat indices for loading
        indices = batch_idx * C * HW + c_idx * HW + hw_idx
        mask_2d = c_mask[:, None] & hw_mask[None, :]
        
        # Load conv output and residual
        conv_val = tl.load(conv_out_ptr + indices, mask=mask_2d, other=0.0)
        residual_val = tl.load(residual_ptr + indices, mask=mask_2d, other=0.0)
        
        # Add
        added = conv_val + residual_val
        
        # Apply batch norm
        normalized = added * scale[:, None] + shift[:, None]
        
        # Accumulate for average pooling (sum over spatial dimension)
        accumulator += tl.sum(normalized, axis=1)
    
    # Compute average
    avg_val = accumulator / HW
    
    # Store output (flattened)
    output_offsets = batch_idx * C + c_offsets
    tl.store(output_ptr + output_offsets, avg_val, mask=c_mask)


@torch.fx.wrap
def fused_add_bn_pool_flatten(conv_out, residual, running_mean, running_var, bn_weight, bn_bias):
    """
    Fused Triton implementation:
    - Add (residual)
    - Batch normalization
    - Adaptive average pooling
    - Flatten
    """
    # Get shapes
    B, C, H, W = conv_out.shape
    HW = H * W
    
    # Prepare output tensor
    output = torch.empty((B, C), dtype=conv_out.dtype, device=conv_out.device)
    
    # Adaptive grid sizing for better GPU utilization
    # Use smaller channel blocks for small batches to increase parallelism
    if B * C < 512:
        # Small batch: use smaller channel blocks for more parallelism
        BLOCK_SIZE_C = min(32, C)
    else:
        # Large batch: use larger blocks for efficiency
        BLOCK_SIZE_C = 64
    
    # Launch fused kernel
    grid = (B, triton.cdiv(C, BLOCK_SIZE_C))
    
    eps = 1e-05
    
    fused_add_bn_pool_kernel[grid](
        conv_out,
        residual,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        output,
        B, C, HW,
        eps=eps,
    )
    
    return output


def replacement_func():
    return fused_add_bn_pool_flatten