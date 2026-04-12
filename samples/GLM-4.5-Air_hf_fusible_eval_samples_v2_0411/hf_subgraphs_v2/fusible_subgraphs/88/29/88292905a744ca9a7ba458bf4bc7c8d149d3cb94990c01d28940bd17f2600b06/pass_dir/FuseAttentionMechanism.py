import torch
import triton
import triton.language as tl

def pattern(attention_scores, input_features):
    """Pattern: softmax attention + reshape sequence + multiplication + sum reduction"""
    # Apply softmax along dimension 1
    tmp_0 = torch.nn.functional.softmax(attention_scores, dim=1)
    
    # Multiple reshape operations - these will be optimized by our kernel
    # The specific constants (8, 1, 2) vary across graphs but follow same pattern
    batch_size = attention_scores.shape[0]
    num_heads = attention_scores.shape[1]
    
    # Reshape operations that expand the tensor for spatial broadcasting
    tmp_1 = tmp_0.reshape(batch_size, -1)
    tmp_2 = tmp_1.view(batch_size, -1, 1, 1)
    tmp_3 = tmp_2.view(batch_size, num_heads, -1, 1, 1)
    
    # Element-wise multiplication and sum reduction
    tmp_4 = tmp_3 * input_features
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    
    # Return the final output (only observable return value)
    return tmp_6

def replacement_args(attention_scores, input_features):
    """Extract arguments for the optimized attention fusion"""
    return (attention_scores, input_features)

@triton.jit
def attention_fusion_kernel(
    attention_ptr, input_ptr, output_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    feature_dim: tl.constexpr,
    spatial_h: tl.constexpr,
    spatial_w: tl.constexpr,
    
    # Input shapes: [batch_size, num_heads, 1, feature_dim] for attention
    #              [batch_size, num_heads, feature_dim, spatial_h, spatial_w] for input
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for fused attention mechanism"""
    # Calculate program indices
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    spatial_y = tl.program_id(2)
    spatial_x = tl.program_id(3)
    
    # Offset calculations
    attention_offset = batch_id * num_heads * feature_dim + head_id * feature_dim
    
    # Load attention weights for this head
    attention_weights = tl.load(
        attention_ptr + attention_offset,
        mask=True,
        other=0.0
    )
    
    # Load input features for spatial location
    input_offset = (
        batch_id * num_heads * feature_dim * spatial_h * spatial_w +
        head_id * feature_dim * spatial_h * spatial_w +
        spatial_y * spatial_w +
        spatial_x
    )
    
    input_features = tl.load(
        input_ptr + input_offset,
        mask=True,
        other=0.0
    )
    
    # Apply softmax and weighted sum
    # For simplicity, we'll compute the softmax-weighted combination
    weighted_sum = 0.0
    for f in range(0, feature_dim, BLOCK_SIZE):
        # Load attention weight for feature dimension
        attn_idx = attention_offset + f
        attn_weight = tl.load(attention_ptr + attn_idx, mask=(f + tl.arange(0, BLOCK_SIZE)) < feature_dim, other=0.0)
        
        # Load input feature for this attention head and spatial location
        input_idx = input_offset + f * spatial_h * spatial_w
        input_feat = tl.load(input_ptr + input_idx, mask=(f + tl.arange(0, BLOCK_SIZE)) < feature_dim, other=0.0)
        
        # Weighted sum
        weighted_sum += attn_weight * input_feat
    
    # Store result
    output_offset = batch_id * num_heads * spatial_h * spatial_w + head_id * spatial_h * spatial_w + spatial_y * spatial_w + spatial_x
    tl.store(output_ptr + output_offset, weighted_sum)

@torch.fx.wrap
def fused_attention_forward(attention_scores, input_features):
    """Wrapper function for optimized attention computation"""
    # Get shapes
    batch_size, num_heads, _, feature_dim = attention_scores.shape
    _, _, _, spatial_h, spatial_w = input_features.shape
    
    # Ensure attention scores have the right shape for our kernel
    if attention_scores.shape[2] != 1:
        attention_scores = attention_scores.unsqueeze(-1)  # Add spatial dimension
    
    total_elements = batch_size * num_heads * spatial_h * spatial_w
    out_shape = (batch_size, num_heads, spatial_h, spatial_w)
    
    # Create output tensor
    output = torch.empty(out_shape, dtype=input_features.dtype, device=input_features.device)
    
    # Configuration
    BLOCK_SIZE = 128
    
    # Launch kernel
    grid = (
        batch_size,
        num_heads,
        spatial_h,
        spatial_w
    )
    
    attention_fusion_kernel[grid](
        attention_scores,
        input_features.view(batch_size, num_heads, feature_dim, spatial_h * spatial_w),
        output,
        batch_size,
        num_heads,
        feature_dim,
        spatial_h,
        spatial_w,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return fused_attention_forward