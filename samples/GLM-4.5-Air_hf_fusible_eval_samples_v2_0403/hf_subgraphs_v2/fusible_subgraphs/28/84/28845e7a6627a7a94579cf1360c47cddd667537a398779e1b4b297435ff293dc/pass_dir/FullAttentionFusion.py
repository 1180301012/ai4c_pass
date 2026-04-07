import torch
from torch import device
import triton
import triton.language as tl
import math

# Pattern matching function - parameterized to handle different view shapes
def pattern(x, y, view_shape):
    """
    Complete fusion: addition + masking + reshape + softmax + dropout
    This matches the entire attention computation pattern
    """
    # Addition operation (adding attention mask to attention scores)
    tmp_0 = x + y
    # Create negative infinity constant and apply max (masking)
    tmp_1 = torch.tensor(-3.4028234663852886e+38, device=device(type='cuda', index=0))
    tmp_2 = torch.max(tmp_0, tmp_1)
    # Reshape to final softmax dimensions
    tmp_3 = tmp_2.view(view_shape)
    # Softmax on last dimension
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    # Dropout with training=False is just scaling by (1-p) = 0.9
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
    return tmp_5

# Full attention fusion kernel
@triton.jit
def full_attention_kernel(
    x_ptr,           # Pointer to attention mask tensor [1, 1, H, W]
    y_ptr,           # Pointer to attention scores tensor [1, C, H, W]
    out_ptr,         # Pointer to output tensor [C, H, W]
    x_elements,      # Total elements in x tensor
    y_elements,      # Total elements in y tensor
    C,               # Number of channels (second dimension)
    H,               # Height (third dimension) 
    W,               # Width (fourth dimension)
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    """
    Complete attention computation kernel:
    1. Add attention mask to attention scores
    2. Apply masking (clamp to -inf for masked positions)
    3. Compute softmax on last dimension
    4. Apply dropout scaling
    """
    # Program indices for parallel processing
    pid_y = tl.program_id(0)  # Process by channel groups
    pid_x = tl.program_id(1)  # Process spatial blocks
    
    # Calculate offsets for attention scores [1, C, H, W]
    offset_y = pid_y * BLOCK_SIZE_Y
    offset_x = pid_x * BLOCK_SIZE_X
    
    # Process one channel at a time for simplicity and correctness
    if offset_y >= C:
        return
    
    # Process spatial [H, W] block
    h_offset = offset_x // W
    w_offset = offset_x % W
    h_base = h_offset * BLOCK_SIZE_X // W
    w_base = w_offset * BLOCK_SIZE_X % W
    
    # Create 2D spatial offsets
    h_offsets = h_base + tl.arange(0, BLOCK_SIZE_X // W)
    w_offsets = w_base + tl.arange(0, BLOCK_SIZE_X)
    
    # Create spatial masks
    h_mask = h_offsets < H
    w_mask = w_offsets < W
    spatial_mask = h_mask[:, None] and w_mask[None, :]
    
    # Load attention scores for current channel [H, W]
    y_idx = (offset_y * H * W + h_offsets[:, None] * W + w_offsets[None, :]).to(tl.int32)
    y_vals = tl.load(y_ptr + y_idx, mask=spatial_mask, other=-float('inf'))
    
    # Load attention mask - broadcast from [1, 1, H, W] to [H, W]
    x_idx = (h_offsets[:, None] * W + w_offsets[None, :]).to(tl.int32)
    x_vals = tl.load(x_ptr + x_idx, mask=spatial_mask, other=0.0)
    
    # Step 1: Add attention mask to scores
    scores = y_vals + x_vals
    
    # Step 2: Apply masking (clamp to -inf)
    masked_scores = tl.where(scores > -3.4028234663852886e+38, scores, -3.4028234663852886e+38)
    
    # Step 3: Compute softmax for each spatial position across all channels
    # Reshape to [H*W, C] for softmax computation
    H_ = tl.sum(h_mask, dtype=tl.int32)
    W_ = tl.sum(w_mask, dtype=tl.int32)
    spatial_size = H_ * W_
    
    # Compute max for numerical stability
    max_scores = tl.max(masked_scores, axis=0)
    
    # Compute exponential
    exp_scores = tl.exp(masked_scores - max_scores[None, :])
    
    # Compute normalization sum
    sum_exp = tl.sum(exp_scores, axis=0)
    
    # Compute softmax
    softmax_scores = exp_scores / sum_exp[None, :]
    
    # Step 4: Apply dropout scaling (multiply by 0.9 when training=False)
    dropout_scale = 0.9
    output = softmax_scores * dropout_scale
    
    # Store result [C, H, W]
    out_idx = (y_idx).to(tl.int32)  # Same indexing as input
    tl.store(out_ptr + out_idx, output, mask=spatial_mask)

@torch.fx.wrap
def full_attention_fusion(x, y, view_shape):
    """
    Complete fusion of attention computation pipeline:
    Addition + Masking + Reshape + Softmax + Dropout
    """
    # Extract dimensions from view_shape
    C, H, W = view_shape
    
    # Ensure tensors are on the same device
    x = x.to(device=y.device)
    
    # Choose optimal block sizes for GPU
    BLOCK_SIZE_X = 1024  # Process spatial elements in large blocks
    BLOCK_SIZE_Y = 32    # Process multiple channels in parallel
    
    # Calculate grid dimensions
    num_channels = C
    num_spatial_blocks = (H * W + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    num_channel_blocks = (num_channels + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    
    # Create output tensor with view shape [C, H, W]
    out = torch.empty((C, H, W), dtype=y.dtype, device=y.device)
    
    # Launch full attention kernel
    full_attention_kernel[(num_channel_blocks, num_spatial_blocks)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        x_elements=x.numel(),
        y_elements=y.numel(),
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )
    
    return out

# Argument extraction function
def replacement_args(x, y, view_shape):
    return (x, y, view_shape)

# Replacement function - returns the full fusion function
def replacement_func():
    return full_attention_fusion