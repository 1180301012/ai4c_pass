import torch
import triton
import triton.language as tl

def pattern(conv_out, cls_token, pos_embed):
    """Pattern: flatten + transpose + expand + cat + add + dropout"""
    tmp_9 = conv_out.flatten(2)
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = cls_token.expand(1, -1, -1)
    tmp_12 = torch.cat([tmp_11, tmp_10], dim=1)
    tmp_13 = tmp_12 + pos_embed
    tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False)
    return tmp_14

def replacement_args(conv_out, cls_token, pos_embed):
    return (conv_out, cls_token, pos_embed)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_TOKEN': 1, 'BLOCK_SIZE_FEAT': 128}),
        triton.Config({'BLOCK_SIZE_TOKEN': 1, 'BLOCK_SIZE_FEAT': 256}),
        triton.Config({'BLOCK_SIZE_TOKEN': 1, 'BLOCK_SIZE_FEAT': 512}),
        triton.Config({'BLOCK_SIZE_TOKEN': 1, 'BLOCK_SIZE_FEAT': 1024}),
        triton.Config({'BLOCK_SIZE_TOKEN': 2, 'BLOCK_SIZE_FEAT': 256}),
        triton.Config({'BLOCK_SIZE_TOKEN': 2, 'BLOCK_SIZE_FEAT': 512}),
        triton.Config({'BLOCK_SIZE_TOKEN': 4, 'BLOCK_SIZE_FEAT': 256}),
        triton.Config({'BLOCK_SIZE_TOKEN': 4, 'BLOCK_SIZE_FEAT': 512}),
        triton.Config({'BLOCK_SIZE_TOKEN': 8, 'BLOCK_SIZE_FEAT': 256}),
    ],
    key=['seq_len', 'hidden_dim'],
)
@triton.jit
def fused_vit_preprocessing_kernel_2d(
    conv_ptr,
    cls_ptr,
    pos_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    seq_len,
    hidden_dim,
    BLOCK_SIZE_TOKEN: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    """
    Fused ViT preprocessing with 2D tiling for better memory access
    conv_out: [B, C, H, W]
    cls_token: [1, 1, hidden_dim]
    pos_embed: [1, seq_len, hidden_dim]
    output: [1, seq_len, hidden_dim] where seq_len = H*W + 1
    """
    pid_token = tl.program_id(0)
    pid_feat = tl.program_id(1)
    
    # Compute token and feature ranges for this block
    token_start = pid_token * BLOCK_SIZE_TOKEN
    feat_start = pid_feat * BLOCK_SIZE_FEAT
    
    feat_offsets = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
    feat_mask = feat_offsets < hidden_dim
    
    spatial_len = height * width
    
    # Process each token in the block
    for i in range(BLOCK_SIZE_TOKEN):
        token_idx = token_start + i
        if token_idx < seq_len:
            # Output offset for this token's features
            out_offsets = token_idx * hidden_dim + feat_offsets
            
            if token_idx == 0:
                # Load cls_token
                data = tl.load(cls_ptr + feat_offsets, mask=feat_mask, other=0.0)
            else:
                # Load from conv_out
                patch_token_idx = token_idx - 1
                # Conv layout: [B, C, H*W]
                # Read position [0, feat_offsets, patch_token_idx]
                conv_offsets = feat_offsets * spatial_len + patch_token_idx
                data = tl.load(conv_ptr + conv_offsets, mask=feat_mask, other=0.0)
            
            # Add position embedding
            pos_data = tl.load(pos_ptr + out_offsets, mask=feat_mask, other=0.0)
            result = data + pos_data
            
            # Store result
            tl.store(output_ptr + out_offsets, result, mask=feat_mask)

@torch.fx.wrap
def fused_vit_preprocessing(conv_out, cls_token, pos_embed):
    """
    Fused ViT preprocessing kernel
    conv_out: [B, C, H, W]
    cls_token: [1, 1, hidden_dim]
    pos_embed: [1, seq_len, hidden_dim]
    Returns: [1, seq_len, hidden_dim]
    """
    batch_size, channels, height, width = conv_out.shape
    hidden_dim = channels
    spatial_len = height * width
    seq_len = spatial_len + 1  # +1 for cls_token
    
    output = torch.empty(1, seq_len, hidden_dim, device=conv_out.device, dtype=conv_out.dtype)
    
    def grid(meta):
        return (
            triton.cdiv(seq_len, meta['BLOCK_SIZE_TOKEN']),
            triton.cdiv(hidden_dim, meta['BLOCK_SIZE_FEAT']),
        )
    
    fused_vit_preprocessing_kernel_2d[grid](
        conv_out,
        cls_token,
        pos_embed,
        output,
        batch_size,
        channels,
        height,
        width,
        seq_len,
        hidden_dim,
    )
    
    return output

def replacement_func():
    return fused_vit_preprocessing