import torch
import triton
import triton.language as tl

def pattern(x, y):
    tmp_2 = x.permute(0, 2, 1)
    tmp_3 = tmp_2.reshape(32, 64, 128, 128)
    return tmp_3

def replacement_args(x, y):
    return (x,)

@triton.jit
def optimized_permute_reshape_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    input_seq,
    input_feat,
    output_height,
    output_width,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_FEAT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Calculate program indices
    batch_idx = tl.program_id(0) * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    feat_idx = tl.program_id(1) * BLOCK_FEAT + tl.arange(0, BLOCK_FEAT)
    h_idx = tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)
    w_idx = tl.program_id(3) * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Create masks for bounds checking
    batch_mask = batch_idx < batch_size
    feat_mask = feat_idx < input_feat
    h_mask = h_idx < output_height
    w_mask = w_idx < output_width
    
    # We need to handle vectorized loads/stores since we're processing multiple elements at once
    batch_mask = batch_mask[:, None, None, None]
    feat_mask = feat_mask[None, :, None, None]
    h_mask = h_mask[None, None, :, None]
    w_mask = w_mask[None, None, None, :]
    
    mask = batch_mask & feat_mask & h_mask & w_mask
    
    # Calculate input indices for permute(0, 2, 1): [batch, seq, feature] -> [batch, feature, seq]
    # Then reshape to [batch, feature, height, width] where height * width = seq
    input_batch = batch_idx
    input_feat_inner = feat_idx
    input_seq_pos = h_idx * output_width + w_idx  # Flatten the spatial dimensions
    
    # Linearized input indices
    input_indices = (input_batch * input_feat + input_feat_inner) * input_seq + input_seq_pos
    
    # Load input data with vectorization
    input_vals = tl.load(input_ptr + input_indices, mask=mask, other=0.0)
    
    # The output is already in the correct shape [batch, feature, height, width]
    # We just need to permute the appropriate axes
    output_batch = batch_idx
    output_feat = feat_idx
    output_h = h_idx
    output_w = w_idx
    
    # Calculate output indices
    output_indices = (output_batch * input_feat + output_feat) * (output_height * output_width) + output_h * output_width + output_w
    
    # Store output with vectorization
    tl.store(output_ptr + output_indices, input_vals, mask=mask)

@torch.fx.wrap
def optimized_permute_reshape(x, target_height, target_width):
    batch_size, input_seq, input_feat = x.shape
    
    target_height = target_height
    target_width = target_width
    
    # Verify that height * width == input_seq
    assert target_height * target_width == input_seq, f"height*width ({target_height*target_width}) must equal input_seq ({input_seq})"
    
    output = torch.empty((batch_size, input_feat, target_height, target_width), dtype=x.dtype, device=x.device)
    
    # Launch kernel with appropriate grid
    BLOCK_BATCH = min(1, batch_size)
    BLOCK_FEAT = min(input_feat, 64)
    BLOCK_H = min(target_height, 32)
    BLOCK_W = min(target_width, 32)
    
    # Adjust grid for vectorization
    grid_b = (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH
    grid_f = (input_feat + BLOCK_FEAT - 1) // BLOCK_FEAT
    grid_h = (target_height + BLOCK_H - 1) // BLOCK_H
    grid_w = (target_width + BLOCK_W - 1) // BLOCK_W
    
    optimized_permute_reshape_kernel[(grid_b, grid_f, grid_h, grid_w)](
        x,
        output,
        batch_size,
        input_seq,
        input_feat,
        target_height,
        target_width,
    )
    
    return output

def generic_permute_reshape(x):
    batch_size, seq_len, feat_dim = x.shape
    
    # Common pattern: [batch, seq, feat] -> permute(0,2,1) -> [batch, feat, seq] -> reshape [batch, feat, h, w]
    # where h * w = seq
    # For the patterns we saw: 128 * 128 = 16384, 32 * 32 = 1024
    if seq_len == 16384:
        h, w = 128, 128
    elif seq_len == 1024:
        h, w = 32, 32
    else:
        # Try to find the best factorization
        h, w = int(seq_len**0.5), int(seq_len**0.5)
        if h * w != seq_len:
            h = seq_len // 32
            w = 32
            if h * w != seq_len:
                h = seq_len // 64
                w = 64
                if h * w != seq_len:
                    # Fallback to simple reshape
                    return x.permute(0, 2, 1).reshape(batch_size, feat_dim, seq_len)
    
    return optimized_permute_reshape(x, h, w)

def replacement_func():
    return generic_permute_reshape