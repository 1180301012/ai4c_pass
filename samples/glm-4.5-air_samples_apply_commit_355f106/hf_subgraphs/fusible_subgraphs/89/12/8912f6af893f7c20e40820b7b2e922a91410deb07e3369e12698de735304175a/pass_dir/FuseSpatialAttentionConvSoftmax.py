import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias):
    # Spatial attention computation pattern: conv1x1 + view + softmax + unsqueeze
    # This matches the sequence: conv2d -> view -> softmax -> unsqueeze
    conv_result = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    reshaped = conv_result.view(conv_result.shape[0], 1, conv_result.shape[2] * conv_result.shape[3])
    softmax_out = torch.nn.functional.softmax(reshaped, 2, _stacklevel=5)
    final_out = softmax_out.unsqueeze(-1)
    return final_out

def replacement_args(conv_input, conv_weight, conv_bias):
    return (conv_input, conv_weight, conv_bias)

@triton.jit
def spatial_attention_kernel(
    x_ptr,           # Input feature map [B, C_in, H, W]
    weight_ptr,      # Conv weight [1, C_in, 1, 1] 
    bias_ptr,        # Conv bias [1]
    out_ptr,         # Output attention weights [B, 1, H*W, 1]
    batch_size,       
    in_channels,     
    in_height,       
    in_width,        
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for spatial attention computation: conv1x1 + softmax + unsqueeze"""
    # Program identifiers
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    
    # Compute spatial position
    spatial_offset = spatial_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offset < (in_height * in_width)
    
    if not tl.any(spatial_mask):
        return
        
    # Reshape spatial index to (h, w) coordinates
    h_idx = spatial_offset // in_width
    w_idx = spatial_offset % in_width
    
    # Load bias (scalar, broadcast to all elements)
    bias = tl.load(bias_ptr)
    
    # Compute conv1x1 output for each spatial location
    # For spatial attention, we collapse input channels to single output channel
    conv_out = tl.zeros([BLOCK_SIZE], dtype=tl.float32) + bias
    
    # Vectorized loading of input channels and weights
    for c in range(0, in_channels, BLOCK_SIZE):
        channel_offset = c + tl.arange(0, tl.minimum(BLOCK_SIZE, in_channels - c))
        
        # Load input slice: [batch_idx, channel_offset, h_idx, w_idx]
        x_ptrs = x_ptr + batch_idx * in_channels * in_height * in_width + \
                 channel_offset[:, None] * in_height * in_width + \
                 h_idx[:, None] * in_width + \
                 w_idx[None, :]
        
        # Load weights: [1, channel_offset, 1, 1] = channel_offset
        weight_ptrs = weight_ptr + channel_offset
        
        x_vals = tl.load(x_ptrs, mask=spatial_mask[:, None], other=0.0)
        weight_vals = tl.load(weight_ptrs, other=0.0)
        
        # Conv1x1: sum over input channels
        conv_out += tl.sum(x_vals * weight_vals[:, None], axis=1)
    
    # Compute max for numerical stability
    max_val = tl.max(conv_out, mask=spatial_mask)
    
    # Subtract max and exponentiate
    exp_vals = tl.exp(conv_out - max_val, mask=spatial_mask)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_vals, mask=spatial_mask)
    
    # Softmax: exp / sum_exp
    softmax_out = exp_vals / sum_exp
    
    # Store output with unsqueeze(-1): reshape spatial results back to original dimensions
    spatial_ptr = out_ptr + batch_idx * (in_height * in_width) * 1 + \
                  spatial_offset * 1
    tl.store(spatial_ptr, softmax_out, mask=spatial_mask)

@torch.fx.wrap
def fused_spatial_attention(conv_input, conv_weight, conv_bias):
    """
    Fused spatial attention computation: conv1x1 + reshape + softmax + unsqueeze
    """
    batch_size, in_channels, in_height, in_width = conv_input.shape
    spatial_locations = in_height * in_width
    
    # Output shape: [batch_size, 1, spatial_locations, 1]  
    out_shape = (batch_size, 1, spatial_locations, 1)
    output = torch.empty(out_shape, dtype=torch.float32, device=conv_input.device)
    
    # Block size for spatial parallelization
    BLOCK_SIZE = 256
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        (spatial_locations + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )
    
    # Launch kernel
    spatial_attention_kernel[grid](
        conv_input,
        conv_weight,
        conv_bias,
        output,
        batch_size,
        in_channels,
        in_height,
        in_width,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_spatial_attention